"""The dispatch floor, and everything `KERNEL_FUN_KDA_MIN_CTAS` must not move with it.

The floor is the one gate in `is_supported` that is a performance heuristic rather than a
capability, and where it falls depends on the workload — so it is configurable, and the
whole risk of that is a knob that quietly relaxes something else. Hence the shape of this
file: one small group for reading the value, and a larger one for the gates that stay
exactly where they were while it moves.

The B300 shape driving it is the small OLMoE3 candidate, B4/HV8/V256 — 128 CTAs, half the
default. Below the floor fla is faster in general; at that shape, measured, it is not.
"""

from __future__ import annotations

import logging

import pytest
import torch

from . import _inputs as I

# B * HV * (V // 64) == 128, the measured shape's grid.
MEASURED: dict = dict(B=4, H=8, HV=8, K=128, V=256)


# --- reading the value: pure, and the only part of this that needs no GPU ---------------


@pytest.mark.parametrize("raw,floor", [("128", 128), ("1", 1), ("4096", 4096), (" 128 ", 128)])
def test_min_ctas_reads_the_environment_per_family(monkeypatch, raw, floor):
    from kernel_fun._common import support

    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", raw)
    assert support.min_ctas("kda") == floor
    # Scoped like the kill switches: one family's knob is not another's.
    assert support.min_ctas("cconv") == support.MIN_CTAS


def test_min_ctas_defaults_and_is_read_per_call(monkeypatch):
    from kernel_fun._common import support

    monkeypatch.delenv("KERNEL_FUN_KDA_MIN_CTAS", raising=False)
    assert support.min_ctas("kda") == support.MIN_CTAS
    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", "128")
    assert support.min_ctas("kda") == 128
    monkeypatch.delenv("KERNEL_FUN_KDA_MIN_CTAS")
    assert support.min_ctas("kda") == support.MIN_CTAS


@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "", "auto"])
def test_a_bad_floor_warns_and_keeps_the_default(monkeypatch, caplog, raw):
    """A typo in a launcher costs 5% throughput, not a 64-GPU job at step 1. Nothing else
    in `support` raises on an environment variable either; the wrapper's job is to fall
    back, loudly."""
    from kernel_fun._common import support

    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", raw)
    with caplog.at_level(logging.WARNING):
        assert support.min_ctas("kda") == support.MIN_CTAS
    assert "KERNEL_FUN_KDA_MIN_CTAS" in caplog.text


# --- the gate itself --------------------------------------------------------------------


@pytest.mark.usefixtures("sm100")
def test_the_floor_gates_the_measured_shape(monkeypatch):
    from kernel_fun.kda import is_supported

    kwargs, _ = I.make(T=256, requires_grad=False, **MEASURED)
    q, v = kwargs["q"], kwargs["v"]

    ok, reason = is_supported(q, v)
    assert not ok and "128 CTAs < 256" in reason and "fla is faster here" in reason

    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", "128")
    assert is_supported(q, v) == (True, None)

    # One CTA short of the configured floor still falls back, and says whose floor it was.
    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", "129")
    ok, reason = is_supported(q, v)
    assert not ok and "128 CTAs < 129" in reason and "configured floor" in reason

    monkeypatch.delenv("KERNEL_FUN_KDA_MIN_CTAS")
    assert is_supported(q, v)[0] is False


@pytest.mark.usefixtures("sm100")
@pytest.mark.parametrize("arg,value", [
    ("chunk_size", 32), ("safe_gate", True), ("cu_seqlens", "packed"),
    ("disable_recompute", True), ("state_v_first", True), ("some_future_flag", True),
])
def test_no_floor_reaches_the_capability_gates(monkeypatch, arg, value):
    """Every check above the floor is a capability or an argument the package does not
    implement. A floor of 1 must not buy any of them."""
    from kernel_fun.kda import is_supported

    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", "1")
    kwargs, _ = I.make(B=1, T=256, H=1, HV=1, V=64, requires_grad=False)  # 1 CTA
    q, v = kwargs["q"], kwargs["v"]
    assert is_supported(q, v)[0] is True, "the floor is not out of the way; test is void"
    assert is_supported(q, v, **{arg: value})[0] is False


@pytest.mark.usefixtures("sm100")
def test_the_kill_switch_and_the_capture_gate_still_win(monkeypatch):
    from kernel_fun._common import support
    from kernel_fun.kda import is_supported

    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", "1")
    kwargs, _ = I.make(B=1, T=256, H=1, HV=1, V=64, requires_grad=False)
    q, v = kwargs["q"], kwargs["v"]

    monkeypatch.setenv("KERNEL_FUN_KDA_DISABLE", "1")
    assert "DISABLE" in is_supported(q, v)[1]
    monkeypatch.delenv("KERNEL_FUN_KDA_DISABLE")

    monkeypatch.setattr(support, "capturing", lambda: True)  # cold shape, no _WARM entry
    assert "capture" in is_supported(q, v)[1]


# --- what actually runs at 128 CTAs -----------------------------------------------------


@pytest.mark.slow
@pytest.mark.usefixtures("sm100")
def test_lowered_floor_matches_fla_at_the_measured_shape(monkeypatch):
    """A performance gate has to be exactly that: the numbers below the default floor are
    the numbers above it. Same reference and same tolerances as test_parity."""
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    from kernel_fun.kda import chunk_kda

    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", "128")
    kwargs, cots = I.make(T=8192, qk_l2norm=True, gate_in_kernel=True, beta_fp32=True,
                          **MEASURED)
    ours = I.run(chunk_kda, kwargs, cots)
    ref = I.run(fla_chunk_kda, kwargs, cots)

    assert set(ours) == set(ref), "different outputs produced"
    bad = []
    for name, got in ours.items():
        assert torch.isfinite(got).all(), f"{name} is not finite"
        r = I.rel_rms(got, ref[name])
        if r > I.TOL[name]:
            bad.append(f"{name} {r:.2e} > {I.TOL[name]}")
    assert not bad, "; ".join(bad)


@pytest.mark.slow
@pytest.mark.usefixtures("sm100")
def test_lowered_floor_engages_our_chain_but_not_the_stages_below_their_own(monkeypatch):
    """Which stages the opt-in actually buys — the question a 5% end-to-end number cannot
    answer. The b1 scan and dhu keep their own 256-CTA floors and stay on fla at this grid;
    that is deliberate, it is what the `is_supported` log line promises, and it is the half
    of the claim a launch witness is the only way to check.

    T=2048 so the intra kernel clears its separate B*(T/64)*HV >= 1024 floor, as production
    does at T=8192 — this test is about the floors, so every other one has to be met.
    """
    from kernel_fun.kda import chunk_kda

    from .test_fallback import OURS, launched_kernels

    monkeypatch.setenv("KERNEL_FUN_KDA_MIN_CTAS", "128")
    kwargs, cots = I.make(T=2048, qk_l2norm=True, gate_in_kernel=True, beta_fp32=True,
                          **MEASURED)
    launched = launched_kernels(lambda: I.run(chunk_kda, kwargs, cots))

    assert launched & set(OURS), "the floor moved but nothing of ours launched"
    assert "kda_cute_b1" not in launched, "the b1 scan's own floor moved with the dispatch one"
    assert "kda_cute_dhu" not in launched, "dhu's own floor moved with the dispatch one"
