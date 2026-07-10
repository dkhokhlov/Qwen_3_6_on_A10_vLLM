"""Integration: LiteLLM proxy endpoints against the LIVE stack (`make start35` / `start`).

Stack-agnostic: derives the served model set from /v1/models rather than hardcoding
27B vs 35B. A chat round-trip COLD-WAKES vLLM if it is idle-stopped, so timeouts are
generous (>= WAKE_TIMEOUT, 300s on this stack). Auth is OFF (no master_key).
Skipped unless :4000 answers (autouse guard in conftest).
"""
import os

import pytest
import requests

pytestmark = pytest.mark.integration

CHAT_TIMEOUT = float(os.environ.get("VLLM_WAKE_TIMEOUT", "360"))


def _models(base: str) -> set:
    r = requests.get(f"{base}/models", timeout=10)
    r.raise_for_status()
    return {m["id"] for m in r.json().get("data", [])}


def _fastest(models: set) -> str:
    return next(n for n in models if "-nothink" in n)


def test_health_liveliness_is_200(litellm_base):
    root = litellm_base.removesuffix("/v1")
    assert requests.get(f"{root}/health/liveliness", timeout=10).status_code == 200


def test_models_advertise_three_flavors(litellm_base):
    names = _models(litellm_base)
    bare = next((n for n in names if "-preserve" not in n and "-nothink" not in n), None)
    assert bare is not None, f"no bare model in {names}"
    assert names == {bare, f"{bare}-preserve", f"{bare}-nothink"}, names


def test_chat_completions_round_trip(litellm_base):
    r = requests.post(
        f"{litellm_base}/chat/completions",
        json={"model": _fastest(_models(litellm_base)),
              "messages": [{"role": "user", "content": "Reply with the single word OK."}],
              "max_tokens": 16, "stream": False},
        timeout=CHAT_TIMEOUT,
    )
    assert r.status_code == 200, r.text[:500]
    assert r.json()["choices"][0]["message"].get("content") is not None


def test_messages_anthropic_path_round_trip(litellm_base):
    """The /v1/messages pass-through (async_pre_request_hook) must ALSO wake + return."""
    root = litellm_base.removesuffix("/v1")
    r = requests.post(
        f"{root}/v1/messages",
        json={"model": _fastest(_models(litellm_base)), "max_tokens": 16,
              "messages": [{"role": "user", "content": "Reply with the single word OK."}]},
        timeout=CHAT_TIMEOUT,
    )
    assert r.status_code == 200, r.text[:500]
    assert r.json().get("content") is not None  # Anthropic shape: top-level content list
