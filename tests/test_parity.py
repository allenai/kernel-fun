"""Do our kernels compute the same thing as fla's, across the arms production exercises?

fla is the reference here rather than an eager oracle, deliberately: it is the thing being
replaced, both ends round the same way, and it lets these run at shapes an O(T) python loop
could not reach. The tolerances are fla's own (tests/ops/test_kda.py), by way of the
research harness.

The arm list is not decorative. `gstrong` and `gate` cover the strong-decay regime where an
exp2 whose argument is not provably <= 0 NaNs in training and nowhere else; `gva` and
`fp16` and the V sweep cover paths the bench ladder dropped; `beta_fp32` and `no_state`
cover the two places the production call differs from every benched row.
"""

from __future__ import annotations

import pytest
import torch

from . import _inputs as I

pytestmark = pytest.mark.usefixtures("sm100")

ARMS = {
    "base": {},
    "gva": dict(H=4, HV=8),
    "gstrong": dict(gate_norm=1 / 16),
    "l2norm": dict(qk_l2norm=True),
    "gate": dict(gate_in_kernel=True),
    "beta_fp32": dict(beta_fp32=True),
    "prod_call": dict(gate_in_kernel=True, qk_l2norm=True, beta_fp32=True),
}

# Each of these changes a compile key (dtype, K or V), so every one pays four fresh
# cute.compile calls AND a full fla Triton autotune — minutes each, against seconds for an
# arm that reuses a shape. They are real coverage (the bench ladder dropped all four), just
# not coverage worth paying for on every run: -m slow.
SHAPE_ARMS = {
    "k64": dict(K=64),
    "v128": dict(V=128),
    "v512": dict(V=512),
    "fp16": dict(dtype=torch.float16),
}


def _both(kwargs, cots):
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    from kernel_fun.kda import chunk_kda

    ours = I.run(chunk_kda, kwargs, cots)
    ref = I.run(fla_chunk_kda, kwargs, cots)
    return ours, ref


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_matches_fla(arm):
    _check(ARMS[arm], arm)


@pytest.mark.slow
@pytest.mark.parametrize("arm", sorted(SHAPE_ARMS))
def test_matches_fla_off_shape(arm):
    _check(SHAPE_ARMS[arm], arm)


def _check(spec, arm):
    kwargs, cots = I.make(**spec)
    ours, ref = _both(kwargs, cots)
    assert set(ours) == set(ref), f"{arm}: different outputs produced"
    bad = []
    for name, got in ours.items():
        assert torch.isfinite(got).all(), f"{arm}: {name} is not finite"
        r = I.rel_rms(got, ref[name])
        if r > I.TOL[name]:
            bad.append(f"{name} {r:.2e} > {I.TOL[name]}")
    assert not bad, f"{arm}: " + ", ".join(bad)


def test_no_state_and_no_final_state():
    """The production call: initial_state=None, output_final_state=False.

    The kernels have no null-state branch — the wrapper supplies a shared zero state — and
    nothing in the research bench ever exercised that, because its h0 is deliberately
    random. `ht` must come back None and `dh0` must not be produced.
    """
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    from kernel_fun.kda import chunk_kda

    kwargs, cots = I.make(gate_in_kernel=True, qk_l2norm=True)
    kwargs["initial_state"] = None
    kwargs["output_final_state"] = False

    def go(fn):
        for t in (kwargs["q"], kwargs["k"], kwargs["v"], kwargs["g"], kwargs["beta"]):
            t.grad = None
        o, ht = fn(**kwargs)
        assert ht is None, "output_final_state=False must return None"
        (o * cots[0]).sum().backward()
        return {"o": o, "dq": kwargs["q"].grad, "dg": kwargs["g"].grad,
                "dbeta": kwargs["beta"].grad, "dv": kwargs["v"].grad}

    ours, ref = go(chunk_kda), go(fla_chunk_kda)
    for name, got in ours.items():
        r = I.rel_rms(got, ref[name])
        assert r <= I.TOL[name], f"{name} {r:.2e} > {I.TOL[name]}"


@pytest.mark.slow
def test_matches_fla_at_length():
    """One long arm: NT=32 chunks, where a per-chunk bug that cancels at NT=8 shows up."""
    kwargs, cots = I.make(B=4, T=2048, gate_in_kernel=True, qk_l2norm=True)
    ours, ref = _both(kwargs, cots)
    for name, got in ours.items():
        r = I.rel_rms(got, ref[name])
        assert r <= I.TOL[name], f"{name} {r:.2e} > {I.TOL[name]}"
