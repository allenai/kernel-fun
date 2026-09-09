"""Shared fixtures. Everything real here needs a B300; nothing here is checkable elsewhere.

The fallback tests are the exception and run anywhere with a GPU, by monkeypatching the
architecture probe — which is why that probe is a module-level cached function rather than
an inlined capability check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: prod-shaped rows; not in the default run")


@pytest.fixture(scope="session")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return "cuda"


@pytest.fixture(scope="session")
def sm100(cuda):
    from kernel_fun._common.support import arch_ok

    if not arch_ok(0):
        cap = torch.cuda.get_device_capability(0)
        pytest.skip(f"needs sm100 (B200/B300); this is sm{cap[0]}{cap[1]}")
    return cuda


@pytest.fixture(autouse=True)
def _reset_probe_caches():
    """Any test that monkeypatches a probe must not leak that into the next one."""
    yield
    from kernel_fun._common import support

    support.arch_ok.cache_clear()
    support.arch_at_least.cache_clear()
    support.has_cute.cache_clear()
    support._logged.clear()
    support._WARM.clear()
