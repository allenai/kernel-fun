"""Input construction, matching the research harness exactly.

Two distributions are deliberate rather than convenient, and both were learned the hard
way:

  beta in (0, 2), not (0, 1). Production runs allow_neg_eigval=True and computes
  2*sigmoid(w_b(x)), and beta > 1 is exactly where the delta rule's (I - beta k k^T)
  factors get negative eigenvalues — which is what conditions the WY triangular inverse.
  Drawn in (0, 1) that inverse is never asked to do the hard version of its job.

  A_log with exp(A_log) uniform in [1, 16], KDA's own init. That is ~11 natural-log units
  of decay per step typical and ~64 in the tail; a gate whose exp2 argument is not provably
  <= 0 reads green at logsigmoid magnitudes and NaNs in training. The `gate_norm=1/16`
  arm covers the same regime for the pre-computed-gate cases.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BT = 64


def make(
    B=8, T=512, H=8, HV=8, K=128, V=256,
    dtype=torch.bfloat16, seed=0, gate_norm=1.0,
    qk_l2norm=False, gate_in_kernel=False, beta_fp32=False, device="cuda",
    requires_grad=True,
):
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*s):
        return torch.rand(*s, generator=gen, device=device, dtype=torch.float32)

    def randn(*s):
        return torch.randn(*s, generator=gen, device=device, dtype=torch.float32)

    q, k = rand(B, T, H, K), rand(B, T, H, K)
    if not qk_l2norm:  # otherwise the op normalizes, and dq/dk are w.r.t. the raw q/k
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    q, k = q.to(dtype), k.to(dtype)
    v = rand(B, T, HV, V).to(dtype)
    beta = randn(B, T, HV).sigmoid() * 2.0
    beta = beta if beta_fp32 else beta.to(dtype)

    A_log = dt_bias = None
    if gate_in_kernel:
        g = randn(B, T, HV, K)
        A_log = (rand(HV) * 15.0 + 1.0).log()
        dt_bias = randn(HV * K) * 0.5
    else:
        g = F.logsigmoid(randn(B, T, HV, K)) / gate_norm

    tensors = [q, k, v, g, beta] + ([A_log, dt_bias] if gate_in_kernel else [])
    if requires_grad:
        for t in tensors:
            t.requires_grad_(True)

    do = randn(B, T, HV, V).to(dtype)
    dht = randn(B, HV, K, V)
    kwargs = dict(
        q=q, k=k, v=v, g=g, beta=beta,
        use_qk_l2norm_in_kernel=qk_l2norm,
        use_gate_in_kernel=gate_in_kernel,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=randn(B, HV, K, V).requires_grad_(requires_grad),
        output_final_state=True,
    )
    return kwargs, (do, dht)


def run(fn, kwargs, cots, leaves=("q", "k", "v", "g", "beta", "A_log", "dt_bias")):
    """One fwd+bwd. Returns {name: tensor} of outputs and gradients."""
    for name in leaves:
        t = kwargs.get(name)
        if t is not None and t.grad is not None:
            t.grad = None
    st = kwargs.get("initial_state")
    if st is not None:
        st.grad = None
    o, ht = fn(**{k: v for k, v in kwargs.items() if v is not None or k == "initial_state"})
    ((o * cots[0]).sum() + (ht * cots[1]).sum()).backward()
    out = {"o": o, "ht": ht}
    for name in (*leaves, "initial_state"):
        t = kwargs.get(name)
        if t is not None and t.requires_grad:
            out[f"d{name}"] = t.grad
    return out


def rel_rms(got, ref) -> float:
    """fla's own metric: rms(ref - got) / (rms(ref) + eps)."""
    got, ref = got.float(), ref.float()
    return (
        (ref - got).pow(2).mean().sqrt() / (ref.pow(2).mean().sqrt() + 1e-8)
    ).item()


# Lifted from the research harness, which lifted them from fla's tests/ops/test_kda.py.
TOL = {
    "o": 0.005, "ht": 0.005,
    "dq": 0.008, "dk": 0.008, "dv": 0.008, "dinitial_state": 0.008,
    "dbeta": 0.02, "dg": 0.02, "dA_log": 0.02, "ddt_bias": 0.02,
}
