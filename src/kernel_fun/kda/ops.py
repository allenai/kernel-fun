"""`chunk_kda` — a drop-in for `fla.ops.kda.chunk_kda`.

Same signature, same return contract, same validation. On a supported call it runs this
package's kernels; on anything else it forwards the call to fla verbatim, which is why the
fallback is bit-identical rather than merely close.

The gate is a WHITELIST. Every argument is either known-supported, handled here, or forces
the fallback — a new fla flag we have never seen degrades to fla instead of being silently
dropped. `is_supported` returns the reason, because the worst failure mode for a kernel
port is not being slow, it is running fla while everyone believes otherwise.
"""

from __future__ import annotations

import logging

import torch

from .._common import support
from . import autograd, chain

__all__ = ["chunk_kda", "is_supported", "warmup"]

log = logging.getLogger(__name__)

# Arguments this package understands. Anything else present and non-default -> fla.
_HANDLED = frozenset({
    "q", "k", "v", "g", "beta", "scale", "initial_state", "output_final_state",
    "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "use_beta_sigmoid_in_kernel",
    "allow_neg_eigval", "lower_bound", "A_log", "dt_bias", "chunk_size",
})
# Present in fla's signature, and each one forces the fallback (see is_supported).
_UNSUPPORTED = frozenset({
    "cu_seqlens", "cu_seqlens_cpu", "cp_context", "safe_gate", "state_v_first",
    "disable_recompute", "return_intermediate_states", "transpose_state_layout",
})


def _warm_sig(q: torch.Tensor, v: torch.Tensor, initial_state, kwargs: dict) -> tuple:
    """Everything that decides which kernels a call launches — shapes, dtypes, device and
    the kernel-selecting flags. Not the stream: see `support.capture_unsupported_reason`."""
    return (
        "kda", tuple(q.shape), tuple(v.shape), q.dtype, v.dtype, q.device.index,
        initial_state is not None,
        bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
        bool(kwargs.get("use_gate_in_kernel", False)),
        bool(kwargs.get("use_beta_sigmoid_in_kernel", False)),
        bool(kwargs.get("allow_neg_eigval", False)),
        kwargs.get("lower_bound") is not None,
    )


def is_supported(
    q: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    **kwargs,
) -> tuple[bool, str | None]:
    """Can this call use our kernels? Returns (ok, reason-if-not).

    The reason strings are meant to end up in a training log verbatim.
    """
    reason = support.common_unsupported_reason(q, "kda")
    if reason is not None:
        return False, reason

    for name in _UNSUPPORTED:
        val = kwargs.get(name)
        if val not in (None, False):
            return False, f"{name}={val!r} is not implemented here"
    for name in kwargs:
        if name not in _HANDLED and name not in _UNSUPPORTED:
            return False, f"unrecognized argument {name!r} (fla may have grown a flag)"

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    if chunk_size != 64:
        return False, f"chunk_size={chunk_size} (only 64 is implemented)"
    if T % 64 != 0:
        return False, f"T={T} is not a multiple of the chunk size"
    if K not in (64, 128):
        return False, f"K={K} (only 64 and 128 are implemented)"
    if V % 64 != 0:
        return False, f"V={V} is not a multiple of 64"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype {q.dtype} (only bf16 and fp16)"
    if not (q.dtype == v.dtype):
        return False, f"mixed dtypes q={q.dtype} v={v.dtype}"
    if initial_state is not None and initial_state.dtype != torch.float32:
        return False, f"initial_state must be fp32, got {initial_state.dtype}"
    # Below this the CuTe scans underfill the GPU and fla is genuinely faster. Not a
    # correctness gate — the kernels would run — so it is worth stating in the log. Where
    # the crossover actually falls is a property of the workload, which is why
    # KERNEL_FUN_KDA_MIN_CTAS can move THIS gate for a shape somebody has timed. It moves
    # nothing else: every check above is a capability, and every stage below keeps its own.
    ctas = B * HV * (V // 64)
    floor = support.min_ctas("kda")
    if ctas < floor:
        why = "fla is faster here" if floor == support.MIN_CTAS else "configured floor"
        return False, f"grid too small ({ctas} CTAs < {floor}); {why}"
    if floor != support.MIN_CTAS:
        caveat = ""
        if floor < support.MIN_CTAS:
            caveat = (
                f"; per-stage floors do not follow it, so the b1 scan and dhu backwards "
                f"are fla's below {support.MIN_CTAS} CTAs"
            )
        support.log_once(
            f"kernel-fun kda: dispatch floor is {floor} CTAs, not the default "
            f"{support.MIN_CTAS} (KERNEL_FUN_KDA_MIN_CTAS){caveat}"
        )
    # Last, so it only speaks for a call that would otherwise be ours.
    needs_grad = torch.is_grad_enabled() and (q.requires_grad or v.requires_grad)
    reason = support.capture_unsupported_reason(
        _warm_sig(q, v, initial_state, kwargs), needs_grad
    )
    if reason is not None:
        return False, reason
    return True, None


@torch.compiler.disable
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context=None,
    **kwargs,
):
    """Drop-in for ``fla.ops.kda.chunk_kda``. See that function for the full argument docs.

    ``@torch.compiler.disable``: the host path drives compiled CuTe objects through ctypes
    pointer writes and a per-layout call cache, and reads
    ``torch.cuda.current_stream().cuda_stream`` — none of which Dynamo can trace. This
    MOVES a graph break rather than adding one: fla's own ``chunk_kda`` is wrapped in
    ``@dispatch('kda')``, which applies ``torch.compiler.disable`` too. Keep every tensor
    operation inside this function for that reason — a ``.float()`` in the caller would
    split a compiled block in two and cost more than these kernels save.
    """
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    fla_kwargs = dict(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, safe_gate=safe_gate, lower_bound=lower_bound,
        disable_recompute=disable_recompute,
        return_intermediate_states=return_intermediate_states,
        state_v_first=state_v_first, cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu, cp_context=cp_context, **kwargs,
    )

    chunk_size = kwargs.get("chunk_size", 64)
    ok, reason = is_supported(
        q, v, chunk_size=chunk_size, initial_state=initial_state,
        # The kernel-selecting flags are part of the capture gate's shape signature, so
        # they must reach is_supported exactly as chunk_kda marks them warm below.
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, lower_bound=lower_bound,
        safe_gate=safe_gate, state_v_first=state_v_first,
        disable_recompute=disable_recompute,
        return_intermediate_states=return_intermediate_states,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, cp_context=cp_context,
        **{k_: v_ for k_, v_ in kwargs.items() if k_ != "chunk_size"},
    )
    if not ok:
        support.log_once(
            f"kernel-fun kda: falling back to fla — {reason}", logging.WARNING
        )
        return fla_chunk_kda(**fla_kwargs)

    # Same validation fla does, so a malformed call fails identically on both paths.
    B, T, H, K = q.shape
    HV = v.shape[2]
    assert q.shape == k.shape, f"q and k must match, got {q.shape} vs {k.shape}"
    assert HV % H == 0, f"HV={HV} must be divisible by H={H}"
    assert g.shape == (B, T, HV, K), f"g must be {[B, T, HV, K]}, got {list(g.shape)}"
    assert beta.shape == (B, T, HV), f"beta must be {[B, T, HV]}, got {list(beta.shape)}"

    A_log, dt_bias = kwargs.get("A_log"), kwargs.get("dt_bias")
    if use_gate_in_kernel:
        assert A_log is not None, "A_log is required when use_gate_in_kernel=True"
    else:
        A_log = dt_bias = None
    if use_beta_sigmoid_in_kernel:
        from fla.ops.common.gate import fused_beta_sigmoid

        beta = fused_beta_sigmoid(beta, scale=2.0 if allow_neg_eigval else 1.0)
    # The intra backward types beta as q's dtype; production computes it in fp32
    # (w_b(x).float().sigmoid()*2). Casting here rather than widening the kernel — beta is
    # bounded in (0, 2) and dbeta carries the loosest tolerance in the op.
    if beta.dtype != q.dtype:
        beta = beta.to(q.dtype)
    if scale is None:
        scale = K ** -0.5

    h0, h0_was_none = initial_state, initial_state is None
    if h0_was_none:
        h0 = chain.zero_state(B, HV, K, v.shape[-1], q.device)

    support.log_versions_once()
    support.log_once(
        f"kernel-fun kda: engaged (B={B} T={T} H={H} HV={HV} K={K} V={v.shape[-1]} "
        f"chunk={chunk_size} l2norm={use_qk_l2norm_in_kernel} gate={use_gate_in_kernel})"
    )
    sig = _warm_sig(q, v, initial_state, dict(
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, lower_bound=lower_bound,
    ))
    o, ht = autograd.apply(
        q, k, v, g, beta, h0, scale, chunk_size, use_qk_l2norm_in_kernel,
        A_log, dt_bias, lower_bound, h0_was_none, sig,
    )
    support.mark_warm(sig, "fwd")  # the backward marks itself, from the Function
    return o.type_as(q), (ht if output_final_state else None)


def warmup(
    *,
    K: int = 128,
    V: int = 256,
    HV: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    use_qk_l2norm: bool = True,
    use_gate_in_kernel: bool = True,
) -> float:
    """Compile every kernel this op needs, so training step 1 does not.

    Four `cute.compile` calls plus fla's Triton autotuning is tens of seconds. Paid inside
    step 1 it looks exactly like a regression — that is what the last port's reported
    "-3% tokens/sec" turned out to be.

    Compile keys carry no B, T or HV, so a tiny shape compiles the production kernels — but
    it has to clear BOTH floors, the chain's (B*HV*(V//64) >= 256) and the intra kernel's
    own (B*(T/64)*HV >= 1024). Miss the second and three of the four CuTe kernels compile
    while the fourth quietly warms up its Triton fallback instead. Also runs the fla
    compatibility probe, so a version mismatch raises here rather than at step 40,000.

    KERNEL_FUN_KDA_MIN_CTAS earns a second pass. Raised, the first shape has to reach the
    new floor or this call falls back and compiles nothing. LOWERED, it puts production on
    a chain this shape never takes — the b1 scan and dhu keep their own 256 floors, so
    below it they are fla's, and fla autotunes them at step 1 unless something warms them
    here. Hence one pass per grid the process will actually dispatch at.

    Returns the elapsed seconds — log it.
    """
    import time

    from .._common.compat import check_fla
    from ._kernels.bwd_intra import _MIN_CTAS as INTRA_MIN_CTAS

    def one_pass(B: int) -> None:
        kw = dict(device=device, dtype=dtype)
        q = torch.randn(B, T, HV, K, **kw, requires_grad=True)
        k = torch.randn(B, T, HV, K, **kw, requires_grad=True)
        v = torch.randn(B, T, HV, V, **kw, requires_grad=True)
        beta = torch.rand(B, T, HV, **kw, requires_grad=True)
        A_log = dt_bias = None
        if use_gate_in_kernel:
            g = torch.randn(B, T, HV, K, **kw, requires_grad=True)
            A_log = torch.rand(HV, device=device, dtype=torch.float32).add(1).log()
            dt_bias = torch.zeros(HV * K, device=device, dtype=torch.float32)
        else:
            g = torch.nn.functional.logsigmoid(
                torch.randn(B, T, HV, K, device=device, dtype=torch.float32)
            ).requires_grad_(True)
        o, ht = chunk_kda(
            q, k, v, g, beta, initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm, use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log, dt_bias=dt_bias,
        )
        (o.float().square().sum() + ht.float().square().sum()).backward()
        torch.cuda.synchronize()

    check_fla()
    T = 1024
    per_b = HV * max(V // 64, 1)
    dispatch = support.min_ctas("kda")
    # The compile pass: the four CuTe kernels, so it clears the default floor (two of them
    # have their own 256) AND the intra kernel's, and a raised dispatch floor on top.
    B = max(
        -(-max(support.MIN_CTAS, dispatch) // per_b),
        -(-INTRA_MIN_CTAS // (HV * (T // 64))),
        1,
    )
    grids = [B]
    # The autotune pass, only for a lowered floor: the fla stages production is about to
    # run there. Nothing NEW of ours compiles at this grid — the compile keys are the same
    # — so what it costs is one small fwd+bwd and what it buys is fla's autotune.
    if dispatch < support.MIN_CTAS:
        B_low = max(-(-dispatch // per_b), 1)
        if B_low < B:
            grids.append(B_low)

    t0 = time.perf_counter()
    torch.manual_seed(0)
    for b in grids:
        one_pass(b)

    # A warmup that silently compiled nothing is worse than none: it hides the cost it was
    # supposed to move, and the run pays it at step 1 anyway.
    from ._kernels import bwd_dhu, bwd_intra, bwd_scan, fwd_state

    empty = [
        m.__name__.rsplit(".", 1)[-1]
        for m in (fwd_state, bwd_scan, bwd_dhu, bwd_intra)
        if not getattr(m, "_COMPILE_CACHE", {})
    ]
    if empty:
        raise RuntimeError(
            f"kernel-fun kda warmup compiled nothing for {empty} — the shape "
            f"(B={B} T={T} HV={HV} K={K} V={V}) did not reach those kernels, so the cost "
            f"this call exists to move is still waiting in step 1. Most likely a CTA floor: "
            f"the chain needs B*HV*(V//64) >= {support.MIN_CTAS} and the intra kernel needs "
            f"B*(T/64)*HV >= {INTRA_MIN_CTAS}."
        )
    return time.perf_counter() - t0
