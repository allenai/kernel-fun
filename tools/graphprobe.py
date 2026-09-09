"""Can this package's ops be CUDA-graph captured, and what does replay save? — the probe
behind kernels/kda/PLAN-fusion.md §7 step 1 (measured 2026-09-08).

    PYTHONPATH=src python tools/graphprobe.py            # on a B300

Per op (kda kernel-fun, kda fla, cconv at the layer's two channel counts), at the 810m
microbatch: eager GPU ms and CPU-enqueue ms for one fwd+bwd, then capture with
`torch.cuda.graph` and time replay, then max|eager - replay| on every output and grad.

Two things this file encodes that cost a day to learn:
  * the static inputs are created ON the capture stream and warmed there. Leaves used by an
    eager forward on the default stream carry gradient-accumulator nodes on that stream;
    if any such graph is alive at capture time the engine syncs the capture stream to the
    legacy stream and the capture dies with "operation would make the legacy stream depend
    on a capturing blocking stream".
  * The package's own capture gate lets a WARM shape through (one eager fwd+bwd at the
    shape first; `support.capture_unsupported_reason`), so nothing is monkeypatched here:
    the probe exercises the shipped gate. Before 2026-09-08 every capture went to fla.
The CPU-enqueue column includes `torch.autograd.grad`'s own per-call floor (~0.55 ms at
these sizes), which a training step pays once per backward, not once per op.
"""
import statistics, time, traceback, torch

B, T, H, K, V = 8, 8192, 16, 128, 256
WARM, ITERS = 3, 20
KW = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True)


def timed(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    gpu, cpu = [], []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); t0 = time.perf_counter(); s.record()
        fn()
        cpu.append((time.perf_counter() - t0) * 1e3); e.record(); torch.cuda.synchronize()
        gpu.append(s.elapsed_time(e))
    return statistics.median(gpu), statistics.median(cpu)


def make_leaves():
    torch.manual_seed(0)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(B, T, H, K, **kw).requires_grad_()
    k = torch.randn(B, T, H, K, **kw).requires_grad_()
    v = torch.randn(B, T, H, V, **kw).requires_grad_()
    g = torch.randn(B, T, H, K, **kw).requires_grad_()
    beta = (torch.rand(B, T, H, device="cuda") * 2).requires_grad_()
    A_log = (torch.rand(H, device="cuda") * 15 + 1).log().requires_grad_()
    dt_bias = (torch.randn(H * K, device="cuda") * 0.5).requires_grad_()
    return q, k, v, g, beta, A_log, dt_bias


def maxdiff(xs, ys):
    return " ".join(f"{(x.float() - y.float()).abs().max().item():.1e}" for x, y in zip(xs, ys))


def kda_arm(name, fn, do):
    print(f"\n== kda {name}  B={B} T={T} H={H}: fwd+bwd, capture on the warmup stream", flush=True)
    try:
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            leaves = make_leaves()

            def run():
                o, _ = fn(*leaves[:5], A_log=leaves[5], dt_bias=leaves[6], **KW)
                return [o] + list(torch.autograd.grad(o, leaves, do))

            for _ in range(WARM):
                out = run()
            del out
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=side):
            out = run()
        torch.cuda.synchronize()
        ref = [t.clone() for t in run()]
        torch.cuda.synchronize()
        gr.replay(); torch.cuda.synchronize()
        print(f"   parity max|eager-replay| o,grads: {maxdiff(ref, out)}", flush=True)
        eg, ec = timed(run)
        gg, gc = timed(gr.replay)
        print(f"   eager GPU {eg:8.3f} ms CPU {ec:7.3f} | graph GPU {gg:8.3f} ms CPU {gc:6.3f} | saved {eg-gg:+.3f} ms GPU", flush=True)
        del gr
    except Exception:
        print("   FAILED:", flush=True); traceback.print_exc()
    torch.cuda.synchronize()


def cconv_arm(D, do_shape=None):
    print(f"\n== cconv kernel-fun D={D}  B={B} T={T}: fwd+bwd, capture on the warmup stream", flush=True)
    try:
        from kernel_fun.cconv import causal_conv1d

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            torch.manual_seed(0)
            x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16).requires_grad_()
            w = ((torch.rand(D, 4, device="cuda") * 2 - 1) * 0.5).requires_grad_()
            dy = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)

            def run():
                y, _ = causal_conv1d(x=x, weight=w, activation="silu")
                return [y] + list(torch.autograd.grad(y, (x, w), dy))

            for _ in range(WARM):
                out = run()
            del out
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=side):
            out = run()
        torch.cuda.synchronize()
        ref = [t.clone() for t in run()]
        torch.cuda.synchronize()
        gr.replay(); torch.cuda.synchronize()
        print(f"   parity max|eager-replay| y,dx,dw: {maxdiff(ref, out)}", flush=True)
        eg, ec = timed(run)
        gg, gc = timed(gr.replay)
        print(f"   eager GPU {eg:8.3f} ms CPU {ec:7.3f} | graph GPU {gg:8.3f} ms CPU {gc:6.3f} | saved {eg-gg:+.3f} ms GPU", flush=True)
        del gr
    except Exception:
        print("   FAILED:", flush=True); traceback.print_exc()
    torch.cuda.synchronize()


if __name__ == "__main__":
    from fla.ops.kda import chunk_kda as fla_kda

    import kernel_fun
    from kernel_fun import cconv, kda
    from kernel_fun._common import support
    from kernel_fun.kda import chunk_kda

    print("kernel-fun", kernel_fun.versions(), flush=True)
    kda.warmup(K=K, V=V, HV=H)
    cconv.warmup(B=B, T=T, D=(2048, 4096))
    do = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    kda_arm("kernel-fun", chunk_kda, do)
    kda_arm("fla", fla_kda, do)
    for D in (2048, 4096):
        cconv_arm(D)
