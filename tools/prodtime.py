"""Time the package against fla at the production shapes — INTEGRATION.md §2's sanity check.

    PYTHONPATH=src python tools/prodtime.py            # on a B300

Not a benchmark harness (that is `python -m loop bench` in the research repo); this is the
ten-line answer to "does the installed package reproduce the numbers its README claims",
run on the node you are about to train on. CUDA-event timing, median of ITERS after a
warmup, one call per iteration. The kda row is the production call (gate + q/k norm in-op,
fp32 beta, no state); the cconv rows are the layer's q/k and v calls, forward alone and
the isolated backward (each side's backward function called directly — see cconv_rows).
"""

from __future__ import annotations

import statistics

import torch

WARMUP, ITERS = 3, 10


def timed(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def kda_row(B=16, T=8192, H=16, K=128, V=256):
    from fla.ops.kda import chunk_kda as fla_kda

    from kernel_fun.kda import chunk_kda

    torch.manual_seed(0)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(B, T, H, K, **kw, requires_grad=True)
    k = torch.randn(B, T, H, K, **kw, requires_grad=True)
    v = torch.randn(B, T, H, V, **kw, requires_grad=True)
    g = torch.randn(B, T, H, K, **kw, requires_grad=True)
    beta = (torch.rand(B, T, H, device="cuda") * 2).requires_grad_(True)
    A_log = (torch.rand(H, device="cuda") * 15 + 1).log().requires_grad_(True)
    dt_bias = (torch.randn(H * K, device="cuda") * 0.5).requires_grad_(True)
    do = torch.randn(B, T, H, V, **kw)

    def step(fn):
        def run():
            o, _ = fn(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias,
                      use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)
            (o * do).sum().backward()
        return run

    ours, ref = timed(step(chunk_kda)), timed(step(fla_kda))
    print(f"kda   prod8192 gate+norm fwd+bwd   fla {ref:7.2f} ms   kernel-fun {ours:7.2f} ms   {ref / ours:.3f}x")


def cconv_rows(B=16, T=8192, W=4):
    """Forward through the public entry points; backward through each side's backward
    function directly. Timing `.backward()` on a loss would add the loss's own elementwise
    passes and the autograd engine's floor (~1.4 ms at this size — more than our kernel),
    which is why the ladder's fwdbwd rows under-report this kernel (its NOTES.md)."""
    from fla.modules.conv.triton import causal_conv1d_bwd as fla_bwd
    from fla.modules.convolution import causal_conv1d as fla_conv

    from kernel_fun.cconv import causal_conv1d
    from kernel_fun.cconv._kernels import strip

    for D in (2048, 4096):
        torch.manual_seed(0)
        x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        w = (torch.rand(D, W, device="cuda") * 2 - 1) * W ** -0.5
        dy = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)

        def fwd(fn):
            def run():
                with torch.no_grad():
                    fn(x=x, weight=w, activation="silu")
            return run

        f_ours, f_ref = timed(fwd(causal_conv1d)), timed(fwd(fla_conv))
        b_ours = timed(lambda: strip.cconv_bwd(x, w, dy))
        b_ref = timed(lambda: fla_bwd(x, dy, None, w, activation="silu"))
        print(f"cconv D={D} fwd            fla {f_ref:7.3f} ms   kernel-fun {f_ours:7.3f} ms   {f_ref / f_ours:.2f}x")
        print(f"cconv D={D} bwd (isolated) fla {b_ref:7.3f} ms   kernel-fun {b_ours:7.3f} ms   {b_ref / b_ours:.2f}x")


if __name__ == "__main__":
    import kernel_fun
    from kernel_fun import cconv, kda  # families are lazy; nothing is imported eagerly

    print("kernel-fun", kernel_fun.versions())
    kda.warmup(K=128, V=256, HV=16)
    cconv.warmup(B=16, T=8192, D=(2048, 4096))
    kda_row()
    cconv_rows()
