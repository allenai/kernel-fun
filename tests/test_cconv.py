"""The cconv family: does it compute fla's causal_conv1d, and does everything else reach fla?

fla is the reference (the thing being replaced; both ends round the same way). Tolerances
are the ladder's — gnorm's budgets on the relative-RMS metric, since fla's own conv tests
use assert_close with dtype tolerances and there is nothing to borrow in these units.

The arm list is the ladder's plus the two things a bench never exercises: a RAGGED T (the
kernels' masked path — every bench case had T a multiple of the segment size, so that
branch had never run before this file), and non-contiguous x (production hands over a
projection output and fla deliberately does not force it contiguous). Everything here runs
on any GPU with Triton; nothing needs sm100.
"""

from __future__ import annotations

import pytest
import torch

from ._inputs import rel_rms

pytestmark = pytest.mark.usefixtures("cuda")

TOL = {"y": 0.005, "dx": 0.008, "dw": 0.02}


def make(B=2, T=512, D=2048, W=4, dtype=torch.bfloat16, wdtype=torch.float32, seed=0,
         layout="contiguous", device="cuda"):
    """Inputs as the ladder's spec.make_inputs builds them: bf16 x, an fp32 weight from
    nn.Conv1d's own U(-1/sqrt(W), 1/sqrt(W)), and a bf16 cotangent."""
    gen = torch.Generator(device=device).manual_seed(seed)
    if layout == "contiguous":
        x = torch.randn(B, T, D, generator=gen, device=device).to(dtype)
    elif layout == "sliced":  # a channel slice of a wider buffer: stride_t > D, stride_d = 1
        x = torch.randn(B, T, D + 64, generator=gen, device=device).to(dtype)[..., :D]
    elif layout == "transposed":  # channels-first storage: stride_d = T, stride_t = 1
        x = torch.randn(B, D, T, generator=gen, device=device).to(dtype).transpose(1, 2)
    else:
        raise ValueError(layout)
    assert x.shape == (B, T, D)
    w = ((torch.rand(D, W, generator=gen, device=device) * 2 - 1) * W ** -0.5).to(wdtype)
    dy = torch.randn(B, T, D, generator=gen, device=device).to(dtype)
    return x, w, dy


def run(fn, x, w, dy, **extra):
    x = x.detach().requires_grad_(True)
    w = w.detach().requires_grad_(True)
    y, state = fn(x=x, weight=w, activation="silu", **extra)
    assert state is None
    (y * dy).sum().backward()
    return {"y": y, "dx": x.grad, "dw": w.grad}


def _both(x, w, dy, **extra):
    from fla.modules.convolution import causal_conv1d as fla_conv

    from kernel_fun.cconv import causal_conv1d

    return run(causal_conv1d, x, w, dy, **extra), run(fla_conv, x, w, dy, **extra)


def _check(arm, x, w, dy):
    ours, ref = _both(x, w, dy)
    bad = []
    for name, got in ours.items():
        assert torch.isfinite(got).all(), f"{arm}: {name} is not finite"
        assert got.dtype == ref[name].dtype, f"{arm}: {name} dtype {got.dtype} vs fla {ref[name].dtype}"
        r = rel_rms(got, ref[name])
        if r > TOL[name]:
            bad.append(f"{name} {r:.2e} > {TOL[name]}")
    assert not bad, f"{arm}: " + ", ".join(bad)


ARMS = {
    "base": dict(),
    "v": dict(D=4096),
    "h8": dict(D=1024),
    "w2": dict(T=2048, W=2),
    "w3": dict(W=3),
    "ragged": dict(T=1000),          # T % 8 != 0: the masked tail, never hit by the bench
    "short": dict(B=3, T=40, D=256),  # T below the minimum segment: one masked segment
    "fp16": dict(dtype=torch.float16),
    "w_bf16": dict(wdtype=torch.bfloat16),
    "sliced_x": dict(layout="sliced"),
    "transposed_x": dict(layout="transposed"),
}

SLOW_ARMS = {
    "prod_qk": dict(B=16, T=8192, D=2048),
    "prod_v": dict(B=16, T=8192, D=4096),
}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_matches_fla(arm):
    _check(arm, *make(**ARMS[arm]))


@pytest.mark.slow
@pytest.mark.parametrize("arm", sorted(SLOW_ARMS))
def test_matches_fla_at_prod(arm):
    _check(arm, *make(**SLOW_ARMS[arm]))


def test_deterministic():
    """dw is a fixed-shape reduction over per-program partials, never atomics. The first
    call is excluded: autotune trials run the kernel once per config."""
    from kernel_fun.cconv import causal_conv1d

    x, w, dy = make(seed=3)
    run(causal_conv1d, x, w, dy)
    first = run(causal_conv1d, x, w, dy)
    second = run(causal_conv1d, x, w, dy)
    for name in first:
        assert torch.equal(first[name], second[name]), f"{name} is not deterministic"


# --------------------------------------------------------------------------------------------
# fallbacks
# --------------------------------------------------------------------------------------------

OURS = ("cconv_fwd_strip", "cconv_bwd_strip")
FLAS = ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel")


def _fallback_cases(x, w):
    D = x.shape[-1]
    return {
        "bias": dict(bias=torch.randn(D, device=x.device)),
        "residual": dict(residual=torch.randn_like(x)),
        "no_activation": dict(activation=None),
        "initial_state": dict(initial_state=torch.zeros(x.shape[0], D, w.shape[1], device=x.device, dtype=x.dtype)),
        "final_state": dict(output_final_state=True),
        "backend_cuda": dict(backend="cuda"),
        "unknown_flag": dict(some_future_fla_flag=True),
    }


FALLBACK_CASES = ("bias", "residual", "no_activation", "initial_state", "final_state",
                  "backend_cuda", "unknown_flag")


@pytest.mark.parametrize("case", FALLBACK_CASES)
def test_unsupported_goes_to_fla_bit_identically(case):
    from fla.modules.convolution import causal_conv1d as fla_conv

    from kernel_fun.cconv import causal_conv1d, is_supported

    x, w, _ = make(B=2, T=256, D=512)
    cases = _fallback_cases(x, w)
    assert set(cases) == set(FALLBACK_CASES)
    extra = cases[case]
    kw = dict(x=x, weight=w, activation="silu")
    kw.update(extra)

    ok, reason = is_supported(**kw)
    assert not ok and reason, f"{case} should be unsupported"

    if case == "backend_cuda":
        return  # fla's CUDA backend may not be installed; the reason is the assertion
    with torch.no_grad():
        ours = causal_conv1d(**kw)
        ref = fla_conv(**kw)
    for a, b in zip(ours, ref):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert torch.equal(a, b), f"{case}: fallback is not bit-identical to fla"


def test_cu_seqlens_goes_to_fla():
    """Packed documents: fla's contract is a flattened [1, total, D] batch."""
    from fla.modules.convolution import causal_conv1d as fla_conv

    from kernel_fun.cconv import causal_conv1d, is_supported

    from .test_fallback import launched_kernels

    x, w, _ = make(B=1, T=512, D=256)
    cu = torch.tensor([0, 128, 320, 512], device=x.device, dtype=torch.int32)
    kw = dict(x=x, weight=w, activation="silu", cu_seqlens=cu)
    ok, reason = is_supported(**kw)
    assert not ok and "cu_seqlens" in reason
    with torch.no_grad():
        names = launched_kernels(lambda: causal_conv1d(**kw))
        ours, _ = causal_conv1d(**kw)
        ref, _ = fla_conv(**kw)
    assert torch.equal(ours, ref)
    assert not [n for n in names if any(o in n for o in OURS)], names


def test_fp32_x_falls_back():
    from kernel_fun.cconv import is_supported

    x, w, _ = make(dtype=torch.float32)
    ok, reason = is_supported(x, w, activation="silu")
    assert not ok and "dtype" in reason


def test_wide_w_falls_back():
    from kernel_fun.cconv import is_supported

    x, w, _ = make(W=5)
    ok, reason = is_supported(x, w, activation="silu")
    assert not ok and "W=5" in reason


def test_disable_env_switch(monkeypatch):
    from kernel_fun.cconv import is_supported

    x, w, _ = make()
    monkeypatch.setenv("KERNEL_FUN_CCONV_DISABLE", "1")
    ok, reason = is_supported(x, w, activation="silu")
    assert not ok and "DISABLE" in reason


def test_pre_hopper_falls_back(monkeypatch):
    from kernel_fun._common import support
    from kernel_fun.cconv import is_supported

    monkeypatch.setattr(support, "arch_at_least", lambda idx, major: False)
    x, w, _ = make()
    ok, reason = is_supported(x, w, activation="silu")
    assert not ok and "sm" in reason


@pytest.mark.usefixtures("sm100")
def test_capture_of_cold_shape_falls_back(monkeypatch):
    """Same contract as kda's: under capture, only a shape that has run eagerly (fwd, and
    bwd if grads are needed) runs our kernels — the autotuner's trial launches cannot be
    captured. Then a real capture of the warm shape replays bit-identically."""
    from kernel_fun._common import support
    from kernel_fun.cconv import causal_conv1d, is_supported

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        x, w, dy = make(seed=21)
        x = x.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        monkeypatch.setattr(support, "capturing", lambda: True)
        ok, reason = is_supported(x, w, activation="silu")
        assert not ok and "capture" in reason

        monkeypatch.setattr(support, "capturing", lambda: False)
        y, _ = causal_conv1d(x=x, weight=w, activation="silu")
        monkeypatch.setattr(support, "capturing", lambda: True)
        ok, reason = is_supported(x, w, activation="silu")
        assert not ok and "backward" in reason, reason
        monkeypatch.setattr(support, "capturing", lambda: False)
        dx, dw = torch.autograd.grad((y * dy).sum(), (x, w))
        monkeypatch.setattr(support, "capturing", lambda: True)
        ok, reason = is_supported(x, w, activation="silu")
        assert ok, reason
        monkeypatch.undo()
        side.synchronize()

    def run():
        y, _ = causal_conv1d(x=x, weight=w, activation="silu")
        return [y, *torch.autograd.grad((y * dy).sum(), (x, w))]

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        replayed = run()
    g.replay()
    torch.cuda.synchronize()
    for name, a, b in zip(("y", "dx", "dw"), (y, dx, dw), replayed):
        assert torch.equal(a, b), f"{name}: replay differs from eager"


def test_launch_witness():
    """Both strip kernels ran and neither fla conv kernel did — the negative half is what
    separates a speedup from a silent fallback."""
    from kernel_fun.cconv import causal_conv1d

    from .test_fallback import launched_kernels

    x, w, dy = make(seed=5)
    names = launched_kernels(lambda: run(causal_conv1d, x, w, dy))
    blob = " ".join(names)
    for expected in OURS:
        assert expected in blob, f"{expected} did not launch"
    for replaced in FLAS:
        assert replaced not in blob, f"{replaced} ran: that is supposed to be ours"


def test_engaged_run_logs_versions_once(caplog):
    """The versions line comes from the entry point, not from the caller's forward.

    olmo-core used to log it from `KimiDeltaAttention.forward`, which cost two graph
    breaks in a compiled block and raised `TypeError: unhashable type: 'dict'` through
    its `lru_cache`-based `log_once` (2026-09-04). Both entry points do it themselves
    now, and they are `torch.compiler.disable`d, so it is free — once per process.
    """
    from kernel_fun.cconv import causal_conv1d

    x, w, dy = make(seed=11)
    with caplog.at_level("INFO", logger="kernel_fun"):
        run(causal_conv1d, x, w, dy)
        run(causal_conv1d, x, w, dy)
    lines = [r.message for r in caplog.records if r.message.startswith("kernel-fun {")]
    assert len(lines) == 1, f"expected exactly one versions line, got {lines}"
    for key in ("torch", "triton", "fla", "device"):
        assert key in lines[0], f"{key} missing from {lines[0]}"


def test_compiles_under_dynamic_shapes():
    """A compiled block that calls this conv must not blow up in Dynamo.

    The 30M mainline ladder died here on 2026-09-04. Undecorated, the entry point let
    Dynamo take our two-argument `autograd.Function` down `trace_backward_graph`, and
    speculating `cconv_bwd` asserted on `dy.stride(0)` returning a `torch.SymInt`. The
    `torch.compiler.disable` on `causal_conv1d` is what keeps the backward eager and out
    of that path; without it this test raises rather than merely running slowly.

    `dynamic=True` is the point — the crash needs a symbolic stride, so a static-shape
    compile of the same code passes and proves nothing. The second shape re-triggers
    tracing the way a real run's changing batch does.
    """
    from kernel_fun.cconv import causal_conv1d

    def block(x, w):
        y, _ = causal_conv1d(x=x, weight=w, activation="silu")
        return (y * 2).sum()

    compiled = torch.compile(block, dynamic=True)
    for T in (512, 640):
        x, w, _ = make(B=2, T=T, D=1024)
        x = x.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        compiled(x, w).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert w.grad is not None and torch.isfinite(w.grad).all()


def test_warmup_autotunes_both_kernels():
    from kernel_fun.cconv import warmup
    from kernel_fun.cconv._kernels import strip

    for k in (strip.cconv_fwd_strip, strip.cconv_bwd_strip):
        k.cache.clear()
    elapsed = warmup(B=2, T=512, D=(1024, 2048))
    for k in (strip.cconv_fwd_strip, strip.cconv_bwd_strip):
        assert k.cache, f"{k.fn.__name__} was not autotuned by warmup"
    assert elapsed < 300, f"warmup took {elapsed:.0f}s"


def test_fla_signature_is_fully_covered():
    """Every parameter fla accepts is either handled or forces a fallback. An fla release
    that grows a flag fails HERE, loudly."""
    import inspect

    from fla.modules.convolution import causal_conv1d as fla_conv

    from kernel_fun.cconv.ops import _HANDLED, _UNSUPPORTED

    fn = inspect.unwrap(fla_conv)
    params = {
        name for name, p in inspect.signature(fn).parameters.items()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    uncovered = params - _HANDLED - _UNSUPPORTED
    assert not uncovered, f"fla.causal_conv1d has parameters kernel-fun does not classify: {sorted(uncovered)}"
