"""The failure modes a numerics test cannot see: aliasing, pinned memory, warmup, drift.

Each of these is a bug this codebase has actually shipped or nearly shipped. The call cache
rewrites a pointer per launch, so a cache-owned output aliases across calls and a benchmark
that makes one call per iteration never notices. The same cache pinned ~778 MiB of the
first call's tensors at a small shape (~24 GiB at production shapes) until it was fixed.
And a compile cost paid inside step 1 reads as a regression.
"""

from __future__ import annotations

import gc

import pytest
import torch

from . import _inputs as I

pytestmark = pytest.mark.usefixtures("sm100")


def _mem() -> int:
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def test_two_calls_do_not_alias():
    """Both results retained, both checked. A cache-owned output would make the first
    equal the second — the exact bug class the call cache invites."""
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    from kernel_fun.kda import chunk_kda

    kw_a, _ = I.make(seed=1, requires_grad=False)
    kw_b, _ = I.make(seed=2, requires_grad=False)
    with torch.no_grad():
        o_a, ht_a = chunk_kda(**kw_a)
        o_b, ht_b = chunk_kda(**kw_b)   # a's results must survive this
        r_a, _ = fla_chunk_kda(**kw_a)
        r_b, _ = fla_chunk_kda(**kw_b)
    assert not torch.equal(o_a, o_b), "two different inputs produced identical outputs"
    assert I.rel_rms(o_a, r_a) <= I.TOL["o"], "the first call's output was overwritten"
    assert I.rel_rms(o_b, r_b) <= I.TOL["o"]
    assert not torch.equal(ht_a, ht_b)


def test_alternating_shapes():
    """A,B,A,B across two layouts: a cache key collision shows up here and nowhere else."""
    from kernel_fun.kda import chunk_kda

    kw_a, _ = I.make(seed=3, requires_grad=False)
    kw_b, _ = I.make(seed=4, HV=16, H=16, requires_grad=False)
    with torch.no_grad():
        first_a, _ = chunk_kda(**kw_a)
        first_b, _ = chunk_kda(**kw_b)
        second_a, _ = chunk_kda(**kw_a)
        second_b, _ = chunk_kda(**kw_b)
    assert torch.equal(first_a, second_a), "same input, different answer after a B call"
    assert torch.equal(first_b, second_b)


def test_deterministic():
    from kernel_fun.kda import chunk_kda

    kw, cots = I.make(seed=5, gate_in_kernel=True)
    first = I.run(chunk_kda, kw, cots)
    second = I.run(chunk_kda, kw, cots)
    for name in first:
        assert torch.equal(first[name], second[name]), f"{name} is not deterministic"


def test_non_default_stream():
    """The stream handle is part of the call key; a second stream must make a new entry
    rather than launch a descriptor built for the first."""
    from kernel_fun.kda import chunk_kda

    kw, _ = I.make(seed=6, requires_grad=False)
    with torch.no_grad():
        base, _ = chunk_kda(**kw)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            other, _ = chunk_kda(**kw)
        s.synchronize()
    assert torch.equal(base, other)


def test_call_cache_does_not_pin_memory():
    """The first call builds the cache entries; nothing of its tensors may outlive them.

    Measured from BEFORE that call — measuring after it puts the leak in the baseline and
    reports a clean zero, which is how this went unnoticed for two weeks.
    """
    from kernel_fun.kda import chunk_kda

    kw, cots = I.make(seed=7)
    I.run(chunk_kda, kw, cots)  # compile; not part of the measurement
    del kw, cots

    base = _mem()
    kw, cots = I.make(seed=8)
    out = I.run(chunk_kda, kw, cots)
    del kw, cots, out
    pinned = _mem() - base

    # The genuine persistent scratch is the per-kernel decay staging: a few hundred KiB at
    # this shape. A leak is two orders of magnitude larger.
    assert pinned < 16 * 2**20, f"{pinned / 2**20:.1f} MiB retained after a dropped call"


def test_no_per_call_growth():
    from kernel_fun.kda import chunk_kda

    kw, cots = I.make(seed=9)
    I.run(chunk_kda, kw, cots)
    del kw, cots
    base = _mem()
    for i in range(8):
        kw, cots = I.make(seed=10 + i)
        out = I.run(chunk_kda, kw, cots)
        del kw, cots, out
    assert _mem() - base < 16 * 2**20, "memory grows per call"


def test_launch_witness():
    """All four CuTe stages plus the three Triton ones ran, and the fla kernels they
    replace did not — including our own full-K wy kernel, which is now the below-the-floor
    fallback for the transposed pair and must not run at a production-shaped grid.

    The negative half is the point: the research harness asserts only that SOME kernel of
    ours launched, so three of four stages could silently fall back and every row would
    still read green.
    """
    from kernel_fun.kda import chunk_kda

    from .test_fallback import launched_kernels

    # T=1024, not the 512 the other tests use: the intra kernel carries its OWN floor of
    # 1024 CTAs (B * T/64 * HV) on top of the chain's 256, and at T=512 with HV=8 it
    # legitimately hands that stage to the Triton fallback. A witness run below a stage's
    # floor asserts a fallback and calls it a failure — which is how this test found the
    # floor in the first place.
    kw, cots = I.make(seed=20, T=1024, gate_in_kernel=True, qk_l2norm=True)
    names = launched_kernels(lambda: I.run(chunk_kda, kw, cots))
    blob = " ".join(names)
    for expected in ("kda_cute_fwd", "kda_cute_b1", "kda_cute_dhu", "kda_cute_intra",
                     "kda_fwd_zero_upper_kernel", "wy_t_main", "wy_t_side"):
        assert expected in blob, f"{expected} did not launch — a stage fell back to fla"
    for replaced in ("chunk_gated_delta_rule_fwd_kernel_h",
                     "chunk_gated_delta_rule_bwd_kernel_dhu",
                     "chunk_kda_bwd_kernel_intra",
                     "wy_dqkg_wide"):
        assert replaced not in blob, f"{replaced} ran: that stage is supposed to be ours"


def test_warmup_compiles_everything():
    from kernel_fun.kda import warmup
    from kernel_fun.kda._kernels import bwd_dhu, bwd_intra, bwd_scan, fwd_state

    for m in (fwd_state, bwd_scan, bwd_dhu, bwd_intra):
        m._COMPILE_CACHE.clear()
    elapsed = warmup(K=128, V=256, HV=16)
    for m in (fwd_state, bwd_scan, bwd_dhu, bwd_intra):
        assert m._COMPILE_CACHE, f"{m.__name__} was not compiled by warmup"
    assert elapsed < 300, f"warmup took {elapsed:.0f}s"


def test_fla_compat_probe():
    """Every fla internal we call, by name and by keyword. Fails on an fla upgrade that
    moved one, instead of letting the move surface as a wrong number."""
    from kernel_fun._common.compat import check_fla, kernel_arg_names

    assert check_fla()
    # The one kernel we launch ourselves, with our own grid and constexprs: a REORDER
    # upstream would feed our arguments to the wrong parameters, so check the set exactly.
    args = set(kernel_arg_names(
        "fla.ops.kda.chunk_intra", "chunk_kda_fwd_kernel_inter_solve_fused"
    ))
    assert {"q", "k", "g", "beta", "Aqk", "Akk"} <= args, sorted(args)


def test_versions_reports_the_stack():
    import kernel_fun

    v = kernel_fun.versions()
    for key in ("torch", "triton", "fla", "cutlass"):
        assert v.get(key) and v[key] != "not installed", f"{key} missing from versions()"
