"""Integration: websearch_interception end-to-end against the LIVE stack.

Replicates the spike probe that resolved upstream
[#29649](https://github.com/BerRIAI/litellm/issues/29649): POST /v1/messages with
Claude Code's native web_search server tool + a time-sensitive question, and assert
Qwen synthesizes a grounded answer via the agentic loop (a recent year in the text),
not a raw dump or an error. Guards the server-side search path that replaced the
client-side DDG MCP.

Skipped unless :4000 answers (autouse guard in conftest) -- run on the box with
`make start35`, then `make test-integration`."""
import os
import re

import pytest
import requests

pytestmark = pytest.mark.integration

CHAT_TIMEOUT = float(os.environ.get("VLLM_WAKE_TIMEOUT", "360"))


def _fastest_model(base: str) -> str:
    r = requests.get(f"{base}/models", timeout=10)
    r.raise_for_status()
    names = {m["id"] for m in r.json().get("data", [])}
    return next(n for n in names if "-nothink" in n)


def test_web_search_synthesizes_a_grounded_answer(litellm_base):
    root = litellm_base.removesuffix("/v1")
    r = requests.post(
        f"{root}/v1/messages",
        json={
            "model": _fastest_model(litellm_base),
            "max_tokens": 512,
            "messages": [{"role": "user", "content":
                "What is the current UTC date? Use the web_search tool, then answer in one sentence."}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        },
        timeout=CHAT_TIMEOUT,
    )
    assert r.status_code == 200, r.text[:500]
    content = r.json().get("content", [])
    text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    # A synthesized grounded answer names a recent year; an empty/raw dump or an error does not.
    assert re.search(r"202[5-9]", text), f"no recent year in synthesized answer: {text[:300]!r}"
