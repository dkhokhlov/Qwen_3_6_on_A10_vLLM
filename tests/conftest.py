"""Shared pytest fixtures for the Qwen3.6-on-A10-vLLM stack regression suite.

Two things make the fast unit suite hermetic:

1. A minimal `litellm` stub injected into sys.modules at import time, BEFORE any test
   imports litellm_callbacks.py. The real `litellm` package only runs inside the proxy
   container (huge dependency); litellm_callbacks.py uses just `litellm.callbacks` (a list)
   and `litellm.integrations.custom_logger.CustomLogger` (a base class to subclass), so a
   tiny stand-in is enough. Guarded so the real package is used if it happens to be
   installed. This MUST run before test modules are imported, hence module top-level.
2. An autouse guard that skips @pytest.mark.integration tests when the live stack is down.
"""
from __future__ import annotations

import sys
import types
import urllib.request

import pytest

# --------------------------------------------------------------------------- #
# 1. litellm stub (must precede any `import litellm_callbacks` in test modules)
# --------------------------------------------------------------------------- #
if "litellm" not in sys.modules:
    litellm = types.ModuleType("litellm")
    litellm.callbacks = []  # Handler instances are appended here at startup

    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:  # minimal stand-in for litellm's CustomLogger base class
        pass

    custom_logger.CustomLogger = CustomLogger
    integrations.custom_logger = custom_logger
    litellm.integrations = integrations

    sys.modules["litellm"] = litellm
    sys.modules["litellm.integrations"] = integrations
    sys.modules["litellm.integrations.custom_logger"] = custom_logger

# --------------------------------------------------------------------------- #
# Integration: endpoints + skip-if-stack-down guard
# --------------------------------------------------------------------------- #
LITELLM_HEALTH = "http://localhost:4000/health/liveliness"
LITELLM_BASE = "http://localhost:4000/v1"
OPENWEBUI_BASE = "http://localhost:3000"


@pytest.fixture(scope="session")
def litellm_base() -> str:
    return LITELLM_BASE


@pytest.fixture(scope="session")
def openwebui_base() -> str:
    return OPENWEBUI_BASE


@pytest.fixture(autouse=True)
def _require_live_stack(request):
    """Skip @pytest.mark.integration tests unless the proxy answers on :4000."""
    if "integration" not in request.keywords:
        return
    try:
        with urllib.request.urlopen(LITELLM_HEALTH, timeout=3) as r:
            if r.status != 200:
                raise RuntimeError(f"non-200: {r.status}")
    except Exception:
        pytest.skip("live stack not reachable at :4000 — start it with `make start35`")
