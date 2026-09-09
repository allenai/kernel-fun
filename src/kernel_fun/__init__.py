"""kernel-fun — CuTe/Triton kernels for linear-attention ops on Blackwell.

Each op family is a subpackage exporting a drop-in for its flash-linear-attention
counterpart, with the same signature and the same return contract:

    from kernel_fun.kda import chunk_kda          # the KDA chunk kernel (CuTe + Triton)
    from kernel_fun.cconv import causal_conv1d    # the KDA short conv + silu (Triton)
    # gdn, gnorm to follow into the same shell

A call the package does not implement — wrong architecture, wrong shape, a flag we have
never seen, a CUDA graph capture of a shape that has not run eagerly yet — is forwarded to
fla verbatim, so installing this can change how fast a model trains but not what it
computes beyond kernel-level rounding.

Nothing is imported eagerly: `import kernel_fun` must stay cheap on a machine with no GPU,
so torch, triton and the CuTe DSL are pulled in by the family that needs them.

Two repos, and changes go to different ones: this package is developed at
github.com/allenai/kernel-fun (the release artifact — and the only route into
OLMo-core), while the kernels are researched in github.com/allenai/kernel-fun-dev (the ladder,
`kernels/<family>/ideas/*`) and cross over by `tools/vendor.py`. A `kernels/...` path in a
comment here is a path in the LADDER. See the pkg repo's README, "Two repos".

Two environment switches, read per call and documented in the README:
    KERNEL_FUN_DISABLE=1        forward everything to fla (per family: _KDA_, _CCONV_, ...)
    KERNEL_FUN_DEBUG=1          log why a call fell back
"""

from __future__ import annotations

__all__ = ["__version__", "versions"]

__version__ = "0.2.0.dev0"


def versions() -> dict[str, str]:
    """Everything that decides what this package computes and how fast.

    Worth logging once per training run: two of these (the CuTe DSL and cuda-python) are
    not pinned by any requirement and arrive with the base image, and no bench row in the
    research repo records them either. When a run is slower than the last one, this is the
    first thing to diff.
    """
    import importlib

    out: dict[str, str] = {"kernel_fun": __version__}
    for name, mod in (
        ("torch", "torch"),
        ("triton", "triton"),
        ("fla", "fla"),
        ("cutlass", "cutlass"),
        ("cuda-python", "cuda.bindings"),
    ):
        try:
            m = importlib.import_module(mod)
            out[name] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            out[name] = "not installed"
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            out["device"] = f"{torch.cuda.get_device_name()} sm{cap[0]}{cap[1]}"
    except Exception:
        pass
    return out
