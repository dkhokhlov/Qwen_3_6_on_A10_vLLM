"""Integration: websearch_interception wiring smoke-test against the LIVE stack.

POST /v1/messages with Claude Code's native web_search server tool + a time-sensitive
question; assert the stack accepts the tool, runs a search, and returns current text
(a recent year). Confirms the server-side search path that replaced the client-side
DDG MCP is wired up and answering 200.

This is a WIRING smoke-test, NOT a faithful [#29649] regression guard: the year
assertion passes for BOTH a synthesized answer AND a raw SearXNG dump (DDG
titles/snippets contain years), so it cannot alone distinguish the agentic loop from
the raw short-circuit. Strengthening it needs a live response capture on the box.

[#29649]: https://github.com/BerRIAI/litellm/issues/29649
Skipped unless :4000 answers (autouse guard in conftest) -- `make start35`,
then `make test-integration`."""
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
    # Wiring check: 200 + a recent year. A raw SearXNG dump also contains years, so this
    # alone does NOT prove synthesis (see docstring); it only confirms the path answers.
    assert re.search(r"202[5-9]", text), f"no recent year in response: {text[:300]!r}"
