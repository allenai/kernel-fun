"""Can this call use our kernels — and if not, exactly why?

Every public entry point in this package answers that question before doing anything, and
delegates to flash-linear-attention when the answer is no. The reason string is the
important half: a silent fallback reads as a correct 1.00x result, and the last time
kernels from this repo landed in training the single most expensive question was "did they
actually run?".

Probes that touch the driver or import cutlass are cached — the previous port called
`import cutlass.cute` on every forward. Probe functions are module-level and cached so a
test can monkeypatch one and clear its cache to exercise the fallback on any GPU.
"""

from __future__ import annotations

import logging
import os
from functools import cache

import torch

log = logging.getLogger(__name__)

# Serial-scan kernels need a full GPU: a one-CTA-per-(chunk, b*hv) kernel at a 64-CTA grid
# lost 1.4ms to fla in the gdn ladder. Below this the fla path is genuinely faster, so the
# gate is a performance decision, not a capability one.
MIN_CTAS = 256

_logged: set[str] = set()
_VERSIONS_KEY = "kernel-fun versions"  # sentinel in _logged, so a test's reset re-arms it


def log_once(message: str, level: int = logging.INFO) -> None:
    if message in _logged:
        return
    _logged.add(message)
    log.log(level, message)


def log_versions_once() -> None:
    """Log `versions()` the first time any family engages.

    This belongs HERE, not in the caller's module: every entry point that calls it is
    already `torch.compiler.disable`d, so the line costs no graph break, whereas the same
    call from a compiled `forward` splits the block (and, in olmo-core, hit `lru_cache` on
    a dict return and raised `TypeError: unhashable type: 'dict'`). Two of these versions
    — the CuTe DSL and cuda-python — are pinned by nothing and ride in with the image,
    so when a run is slower than the last one this is the first thing to diff.
    """
    if _VERSIONS_KEY in _logged:
        return  # before versions(): it queries device properties, ~14 us, and this is per call
    _logged.add(_VERSIONS_KEY)
    from .. import versions

    log.info(f"kernel-fun {versions()}")


@cache
def has_cute() -> bool:
    """Is the CuTe DSL importable? Cached: the import pulls MLIR and costs seconds.

    Also warns, once, when the DSL's CUDA build visibly disagrees with torch's — see
    `cute_cuda_mismatch`. A warning and not a gate: the wrong-build DSL may well still
    compile (a CUDA 12 toolchain on a CUDA 13 driver is a supported combination), and
    silently routing a working install to fla is the failure this package exists to avoid.
    """
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except Exception:  # pragma: no cover - environment-dependent
        return False
    reason = cute_cuda_mismatch(torch.version.cuda, _installed_dists())
    if reason is not None:
        log_once(f"kernel-fun: {reason}", logging.WARNING)
    return True


def _installed_dists() -> frozenset[str]:
    import importlib.metadata as md

    return frozenset(
        d.metadata["Name"].lower() for d in md.distributions() if d.metadata["Name"]
    )


def cute_cuda_mismatch(torch_cuda: str | None, installed: frozenset[str]) -> str | None:
    """Is the installed CuTe DSL the CUDA build torch was built for? Pure, for the test.

    On PyPI `nvidia-cutlass-dsl` >= 4.6 always pulls `nvidia-cutlass-dsl-libs-cu12` and
    ships the CUDA 13 libraries only through its `[cu13]` extra, so a CUDA 13 torch next to
    a DSL whose cu13 libs are absent means somebody wrote the bare requirement (this
    package's own extra is `kernel-fun[cu13]`; OLMo-core's is FA4's `[cu13]`). Only that
    one direction is detectable: before 4.6 the CUDA 12 libraries live inside the base
    wheel, and a cu13-only install on a CUDA 12 torch cannot be told from a pre-split one.
    """
    if not torch_cuda or "nvidia-cutlass-dsl" not in installed:
        return None
    major = torch_cuda.split(".")[0]
    has_cu12 = "nvidia-cutlass-dsl-libs-cu12" in installed
    has_cu13 = "nvidia-cutlass-dsl-libs-cu13" in installed
    if major == "13" and has_cu12 and not has_cu13:
        return (
            f"torch is a CUDA {torch_cuda} build but the CuTe DSL installed is the CUDA 12 "
            f"one (nvidia-cutlass-dsl-libs-cu12 without -cu13). Install "
            f"nvidia-cutlass-dsl[cu13] — kernel-fun[cu13] does — or expect cute.compile "
            f"to fail rather than fall back"
        )
    return None


@cache
def arch_at_least(device_index: int, major: int) -> bool:
    """For the Triton-only families: any Blackwell-or-newer datacenter part, or Hopper.

    Those kernels need no tcgen05 and no CuTe DSL, so the sm100 gate below would deny them
    for no reason; but nobody has timed them below sm100, so callers state their floor.
    """
    return torch.cuda.get_device_capability(device_index)[0] >= major


@cache
def arch_ok(device_index: int) -> bool:
    """sm100 exactly — Blackwell datacenter (B200/B300).

    Not `major >= 10`: sm_120 is consumer Blackwell and has no tcgen05, so the MMA kernels
    would fail inside cute.compile rather than fall back. Not `major == 10 and minor == 0`
    either: B300 reports (10, 3).
    """
    return torch.cuda.get_device_capability(device_index)[0] == 10


def disabled(family: str) -> bool:
    """KERNEL_FUN_DISABLE=1 kills every family; KERNEL_FUN_<FAMILY>_DISABLE=1 kills one.

    Read per call, never at import: the point of a kill switch is that someone can set it
    on a run that is already failing, and an import-time read would depend on which module
    got imported first.
    """
    return (
        os.environ.get("KERNEL_FUN_DISABLE", "0") == "1"
        or os.environ.get(f"KERNEL_FUN_{family.upper()}_DISABLE", "0") == "1"
    )


def debug() -> bool:
    return os.environ.get("KERNEL_FUN_DEBUG", "0") == "1"


def min_ctas(family: str) -> int:
    """`family`'s dispatch floor: `MIN_CTAS`, or `KERNEL_FUN_<FAMILY>_MIN_CTAS` if set.

    A performance heuristic, not a capability — the kernels run below it, they just lose to
    fla where the CuTe scans underfill the GPU — and the crossover belongs to the workload,
    not to the hardware. So a shape that has been *measured* below the default can opt in
    from outside, instead of a training script assigning to `MIN_CTAS` and moving whichever
    per-stage gates happened to bind it at import.

    Only the chain-level gate moves. Every kernel keeps its own floor, so a lowered value
    changes which stages are ours, never what any of them computes.

    Read per call, like `disabled`. A value that is not a positive integer is ignored with a
    warning: falling back is this package's job, ending someone's run at step 1 is not.
    """
    raw = os.environ.get(f"KERNEL_FUN_{family.upper()}_MIN_CTAS")
    if raw is None:
        return MIN_CTAS
    try:
        floor = int(raw)
    except ValueError:
        floor = 0
    if floor < 1:
        log_once(
            f"kernel-fun {family}: ignoring KERNEL_FUN_{family.upper()}_MIN_CTAS={raw!r} "
            f"(want a positive integer); using the default floor of {MIN_CTAS} CTAs",
            logging.WARNING,
        )
        return MIN_CTAS
    return floor


def capturing() -> bool:
    """Is a CUDA graph being captured on this stream?"""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # pragma: no cover - older torch without the query
        return False


# Call signatures (per family: shapes, dtypes, device, kernel-selecting flags — see the
# family's `_warm_sig`) that have completed an eager forward / backward through our kernels.
_WARM: set[tuple] = set()


def mark_warm(sig: tuple, phase: str) -> None:
    """Record that `sig` just ran `phase` ("fwd" or "bwd") outside any capture."""
    if not capturing():
        _WARM.add((*sig, phase))


def capture_unsupported_reason(sig: tuple, needs_grad: bool) -> str | None:
    """Under CUDA graph capture, our kernels run only for a shape that has already run eagerly.

    Capture bakes every pointer the region touches; the call cache's launch-time pointer
    poke is no different from what cuBLAS does, and with the static buffers a graphed
    callable owns it replays correctly (verified bit-identical fwd+bwd, 2026-09-08). What
    CANNOT happen inside a capture is a first call at a shape: `cute.compile`, and the
    Triton autotuners' trial launches (ours and fla's both synchronize). Both are keyed on
    shapes and dtypes, never on the stream, so a call-cache miss on a new stream for a warm
    shape is only `from_dlpack` plus an allocation, and captures fine. Hence the stream is
    deliberately NOT part of the signature: `make_graphed_callables` warms on one side
    stream and captures on another, and keying on it would send every graphed call to fla
    with nothing but a log line to say so.

    The fallback for a cold shape is fla, which under capture has the same autotune
    problem — but that is fla's failure to report, loudly, not ours to hide.
    """
    if not capturing():
        return None
    if (*sig, "fwd") not in _WARM:
        return (
            "CUDA graph capture of a shape that has not run eagerly yet (run one forward "
            "at this shape before capturing — the compile and autotune steps cannot be "
            "captured)"
        )
    if needs_grad and (*sig, "bwd") not in _WARM:
        return (
            "CUDA graph capture of a shape whose backward has not run eagerly yet (run one "
            "fwd+bwd at this shape before capturing)"
        )
    return None


def _basic_unsupported_reason(t: torch.Tensor, family: str) -> str | None:
    if disabled(family):
        return f"KERNEL_FUN_{family.upper()}_DISABLE / KERNEL_FUN_DISABLE is set"
    if not t.is_cuda:
        return "not a CUDA tensor"
    # CUDA graph capture is gated per shape, after the shape checks: capture_unsupported_reason.
    return None


def common_unsupported_reason(t: torch.Tensor, family: str) -> str | None:
    """The gates every CuTe family shares. Family-specific shape checks live with the family."""
    reason = _basic_unsupported_reason(t, family)
    if reason is not None:
        return reason
    if not arch_ok(t.device.index or 0):
        cap = torch.cuda.get_device_capability(t.device.index or 0)
        return f"device capability sm{cap[0]}{cap[1]} is not sm100 (B200/B300)"
    if not has_cute():
        return "the CUTLASS CuTe DSL is not installed"
    return None


def triton_unsupported_reason(t: torch.Tensor, family: str, min_major: int = 9) -> str | None:
    """The same, for a family whose kernels are all Triton: no CuTe, a lower arch floor."""
    reason = _basic_unsupported_reason(t, family)
    if reason is not None:
        return reason
    if not arch_at_least(t.device.index or 0, min_major):
        cap = torch.cuda.get_device_capability(t.device.index or 0)
        return (
            f"device capability sm{cap[0]}{cap[1]} is below sm{min_major}0 "
            f"(measured on sm100; untimed below it)"
        )
    return None
