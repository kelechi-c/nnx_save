"""Shared pytest configuration for the nnx_save verification suite.

The suite simulates a TPU-style multi-device setup on CPU: XLA is told to
expose 8 host devices so sharded parameters can be exercised without a
TPU/GPU.  This must happen before JAX initialises its backend, therefore it
lives in conftest.py (imported before any test module).
"""

import os
import sys

_EXTRA_DEVICES = "--xla_force_host_platform_device_count=8"
_flags = os.environ.get("XLA_FLAGS", "")
if _EXTRA_DEVICES not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " " + _EXTRA_DEVICES).strip()

# Make the in-repo package importable regardless of how pytest was invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line("markers", "hazard: asserts the safe behaviour for a known silent-failure mode")
    config.addinivalue_line("markers", "tpu_style: multi-device / bf16 / sharding tests")
