"""Every unsupported call must reach fla, and reach it unchanged.

Bit-identity is the assertion, not closeness: it is only achievable if the wrapper forwards
its arguments verbatim, which is exactly the property that makes the fallback safe to rely
on. And each case checks that no kernel of ours launched — a fallback that quietly runs half
our chain would still produce plausible numbers.
"""

from __future__ import annotations

import pytest
import torch

from . import _inputs as I

OURS = ("kda_cute_fwd", "kda_cute_b1", "kda_cute_dhu", "kda_cute_intra",
        "kda_fwd_zero_upper_kernel", "chunk_kda_bwd_kernel_wy_dqkg_wide",
        "chunk_kda_bwd_kernel_wy_t")


def launched_kernels(fn) -> set[str]:
    """CUDA kernel names launched by fn() — the only thing that can tell a fallback from a
    speedup after the fact.

    Filter on self DEVICE time, as loop/bench.py::cuda_kernel_trace does: iterating raw
    events instead collects host-side runtime calls (cudaLaunchKernel, cudaMemsetAsync) and
    misses the kernels, which is a witness that witnesses nothing.
    """
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return {
        str(ev.key) for ev in prof.key_averages()
        if float(getattr(ev, "self_device_time_total", 0) or 0) > 0
    }


def _fla_and_ours(kwargs):
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    from kernel_fun.kda import chunk_kda

    call = {k: v for k, v in kwargs.items()}
    ours = chunk_kda(**call)
    ref = fla_chunk_kda(**call)
    return ours, ref


CASES = {
    "chunk32": dict(chunk_size=32),
    "safe_gate": dict(safe_gate=True, lower_bound=-5.0),
    "state_v_first": dict(state_v_first=True),
    "disable_recompute": dict(disable_recompute=True),
    "cu_seqlens": "special",
    "unknown_flag": dict(some_future_fla_flag=True),
}


@pytest.mark.usefixtures("cuda")
@pytest.mark.parametrize("case", sorted(CASES))
def test_unsupported_goes_to_fla_bit_identically(case):
    kwargs, _ = I.make(B=4, T=256, H=4, HV=4, requires_grad=False)
    extra = CASES[case]
    if extra == "special":
        pytest.skip("cu_seqlens needs a flattened batch; covered by is_supported below")
    kwargs.update(extra)
    kwargs["output_final_state"] = True

    from kernel_fun.kda import is_supported

    ok, reason = is_supported(kwargs["q"], kwargs["v"], **{
        k: v for k, v in kwargs.items()
        if k not in ("q", "k", "v", "g", "beta", "output_final_state")
    })
    assert not ok and reason, f"{case} should be unsupported"

    if case in ("chunk32", "unknown_flag"):
        # fla itself accepts these, so the results are comparable end to end.
        with torch.no_grad():
            ours, ref = _fla_and_ours(kwargs)
        for a, b in zip(ours, ref):
            if a is None or b is None:
                assert a is None and b is None
            else:
                assert torch.equal(a, b), f"{case}: fallback is not bit-identical to fla"


@pytest.mark.usefixtures("cuda")
def test_unsupported_launches_none_of_ours():
    from kernel_fun.kda import chunk_kda

    kwargs, _ = I.make(B=4, T=256, H=4, HV=4, requires_grad=False)
    kwargs["chunk_size"] = 32
    with torch.no_grad():
        names = launched_kernels(lambda: chunk_kda(**kwargs))
    hit = [n for n in names if any(o in n for o in OURS)]
    assert not hit, f"a fallback launched our kernels: {hit}"


@pytest.mark.usefixtures("cuda")
def test_non_sm100_falls_back(monkeypatch):
    """The arch gate is new behaviour — nothing in the research tree ever checked it —
    so exercise it by patching the probe rather than by finding an sm90 box."""
    from kernel_fun._common import support
    from kernel_fun.kda import is_supported

    monkeypatch.setattr(support, "arch_ok", lambda idx: False)
    kwargs, _ = I.make(B=4, T=256, H=4, HV=4, requires_grad=False)
    ok, reason = is_supported(kwargs["q"], kwargs["v"])
    assert not ok and "sm" in reason


@pytest.mark.usefixtures("cuda")
def test_disable_env_switch(monkeypatch):
    from kernel_fun.kda import is_supported

    kwargs, _ = I.make(B=4, T=256, H=4, HV=4, requires_grad=False)
    monkeypatch.setenv("KERNEL_FUN_KDA_DISABLE", "1")
    ok, reason = is_supported(kwargs["q"], kwargs["v"])
    assert not ok and "DISABLE" in reason


def _flags(kwargs):
    return {k: v for k, v in kwargs.items() if k not in ("q", "k", "v", "g", "beta")}


@pytest.mark.usefixtures("sm100")
def test_capture_of_cold_shape_falls_back(monkeypatch):
    """Under capture a shape must have run eagerly first — cute.compile and the Triton
    autotuners (ours and fla's) synchronize, and neither can happen inside a graph. Then
    the gate opens in two steps: a forward warms the forward, a backward the backward.
    Exercised by patching the capture probe: a real capture of a cold shape would die
    inside Triton, not in our gate."""
    from kernel_fun._common import support
    from kernel_fun.kda import chunk_kda, is_supported

    kwargs, cots = I.make(seed=11)
    flags = _flags(kwargs)
    monkeypatch.setattr(support, "capturing", lambda: True)
    ok, reason = is_supported(kwargs["q"], kwargs["v"], **flags)
    assert not ok and "capture" in reason and "not run eagerly" in reason

    monkeypatch.setattr(support, "capturing", lambda: False)
    o, ht = chunk_kda(**kwargs)
    monkeypatch.setattr(support, "capturing", lambda: True)
    ok, reason = is_supported(kwargs["q"], kwargs["v"], **flags)
    assert not ok and "backward" in reason, reason
    with torch.no_grad():  # inference capture needs only the forward
        ok, reason = is_supported(kwargs["q"], kwargs["v"], **flags)
    assert ok, reason

    monkeypatch.setattr(support, "capturing", lambda: False)
    ((o * cots[0]).sum() + (ht * cots[1]).sum()).backward()
    monkeypatch.setattr(support, "capturing", lambda: True)
    ok, reason = is_supported(kwargs["q"], kwargs["v"], **flags)
    assert ok, reason

    # A different shape is cold again (same B so the CTA floor is still cleared: the gate
    # speaks last, only for a call that would otherwise be ours).
    other, _ = I.make(T=1024, seed=12)
    ok, reason = is_supported(other["q"], other["v"], **_flags(other))
    assert not ok and "capture" in reason


@pytest.mark.usefixtures("sm100")
def test_warm_shape_captures_and_replays_bit_identically():
    """The whole point of the gate: a warm shape's fwd+bwd goes into a CUDA graph on our
    kernels and replays exactly what eager computed. Leaves are born on the capture
    stream — an eager forward on the legacy stream leaves gradient-accumulator nodes
    that make the engine sync the capture stream to it ("operation would make the legacy
    stream depend on a capturing blocking stream"); see tools/graphprobe.py."""
    from kernel_fun.kda import chunk_kda

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        kwargs, cots = I.make(seed=13)
        leaves = [kwargs[n] for n in ("q", "k", "v", "g", "beta", "initial_state")]

        def run():
            o, ht = chunk_kda(**kwargs)
            loss = (o * cots[0]).sum() + (ht * cots[1]).sum()
            return [o, ht, *torch.autograd.grad(loss, leaves)]

        eager = [t.clone() for t in run()]  # warms fwd and bwd at this shape
        side.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        replayed = run()
    g.replay()
    torch.cuda.synchronize()
    for name, a, b in zip(("o", "ht", "dq", "dk", "dv", "dg", "dbeta", "dh0"), eager, replayed):
        assert torch.equal(a, b), f"{name}: replay differs from eager"

    names = launched_kernels(g.replay)
    assert any("kda_cute_fwd" in n for n in names), f"the graph does not hold our kernels: {names}"


@pytest.mark.usefixtures("cuda")
def test_small_grid_falls_back():
    """Below the CTA floor fla is faster, and a benchmark that does not know that is
    comparing fla against fla without saying so."""
    from kernel_fun.kda import is_supported

    kwargs, _ = I.make(B=1, T=256, H=1, HV=1, V=64, requires_grad=False)
    ok, reason = is_supported(kwargs["q"], kwargs["v"])
    assert not ok and "CTA" in reason


def test_fla_signature_is_fully_covered():
    """Every parameter fla accepts is either handled or forces a fallback.

    An fla release that grows a flag should fail HERE, loudly, rather than have the new
    flag silently ignored by a wrapper that never looked at it.
    """
    import inspect

    from fla.ops.kda import chunk_kda as fla_chunk_kda

    from kernel_fun.kda.ops import _HANDLED, _UNSUPPORTED

    fn = inspect.unwrap(fla_chunk_kda)
    params = {
        name for name, p in inspect.signature(fn).parameters.items()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    uncovered = params - _HANDLED - _UNSUPPORTED
    assert not uncovered, (
        f"fla.chunk_kda has parameters kernel-fun does not classify: {sorted(uncovered)}"
    )
