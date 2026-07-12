"""Integration: websearch_interception agentic-synthesis guard against the LIVE stack.

POST /v1/messages with Claude Code's native web_search server tool + a time-sensitive
question and assert Qwen SYNTHESIZES a grounded answer (prose), not a raw SearXNG dump.

Request shape selects the path (verified on this box 2026-07-12):
  - web_search ONLY -> LiteLLM [#29649] short-circuit: one text block that is a verbatim
    SearXNG "Title:/URL:/Snippet:" dump, with usage tokens 0/0 -- Qwen never runs. Claude
    Code never sends this shape (it always carries web_search alongside its other
    tools), so the short-circuit never fires for real traffic.
  - web_search + a 2nd tool -> forwarded to vLLM: Qwen reasons over the search results and
    emits a one-sentence answer (stop_reason=end_turn, usage tokens > 0).

This test replicates Claude Code's real shape (web_search + an irrelevant function tool
the model has no reason to call) to exercise the agentic synthesizing loop, then asserts
synthesis on a COMPOSITE of signals -- no single one suffices: a non-empty text block,
the LLM actually ran (usage input/output > 0; the short-circuit is 0/0), stop_reason=
end_turn, and NO dump-like Title:/Snippet:/http repetition in the answer. The year-regex
is kept only as a weak recency signal; it passes for BOTH a synthesized answer AND a raw
dump (DDG titles carry years), so it must never be the sole assertion.

[#29649]: https://github.com/BerRIAI/litellm/issues/29649
Skipped unless :4000 answers (autouse guard in conftest) -- `make start35`,
then `make test-integration`."""
import os
import re

import pytest
import requests

pytestmark = pytest.mark.integration

# The agentic loop needs vLLM up; the lifecycle suite idles/kills it first, so this
# request often cold-wakes the backend (multi-GB reload) AND waits on SearXNG (engines
# can rate-limit). 600s matches the proxy's own request_timeout; the prior 360s was
# ample for the old short-circuit path (which never woke vLLM) but times out here.
CHAT_TIMEOUT = float(os.environ.get("VLLM_WAKE_TIMEOUT", "600"))


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
            # The agentic loop emits a (variable-length) thinking block before the
            # answer; the websearch_interception path reasons even under -nothink, and
            # conflicting snippets can make that reasoning run long. 8192 leaves large
            # headroom over the ~800-token case so the turn reaches end_turn w/ a text
            # answer rather than truncating mid-thinking (which yields no text block).
            "max_tokens": 8192,
            "messages": [{"role": "user", "content":
                "What is the current UTC date? Use the web_search tool, then answer in one sentence."}],
            # Claude Code carries web_search alongside its other tools; that multi-tool
            # shape routes to the agentic loop, NOT the web-search-only #29649 short-
            # circuit. The 2nd tool is deliberately irrelevant + unattractive (required
            # numeric params the model has no values for) so Qwen answers via web_search
            # and stops, rather than calling it.
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "function", "name": "calculate_mortgage",
                 "description": "Compute a monthly mortgage payment given principal, annual interest rate, and term in years.",
                 "input_schema": {"type": "object",
                    "properties": {"principal": {"type": "number"},
                                   "annual_rate": {"type": "number"},
                                   "years": {"type": "integer"}},
                    "required": ["principal", "annual_rate", "years"]}},
            ],
        },
        timeout=CHAT_TIMEOUT,
    )
    assert r.status_code == 200, r.text[:500]
    body = r.json()
    content = body.get("content", [])
    text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    # Synthesis, not a raw SearXNG dump -- composite check (no single signal suffices):
    #  - a non-empty answering text block exists
    #  - the LLM actually ran: the #29649 short-circuit returns usage 0/0
    #  - the turn completed (not truncated mid-reasoning)
    #  - no dump-like repetition: a synthesized sentence carries no Title:/Snippet:
    #    labels or raw http URLs; a SearXNG dump has ~30 of each. <2 tolerates a single
    #    legitimately-quoted URL while still rejecting the dump.
    assert any(b.get("type") == "text" and b.get("text", "").strip() for b in content), \
        f"no non-empty text block: {content!r}"
    usage = body.get("usage", {})
    assert usage.get("input_tokens", 0) > 0 and usage.get("output_tokens", 0) > 0, \
        f"LLM did not run (usage={usage}); likely the #29649 short-circuit raw dump"
    assert body.get("stop_reason") == "end_turn", \
        f"stop_reason={body.get('stop_reason')!r} (expected end_turn): {text[:300]!r}"
    assert text.count("Title:") < 2 and text.count("Snippet:") < 2, \
        f"SearXNG dump labels in answer: {text[:300]!r}"
    assert len(re.findall(r"https?://", text)) < 2, \
        f"raw URL dump in answer: {text[:300]!r}"
    # Weak recency signal only -- passes for both synthesis and a dump, so it is never
    # the sole gate (see docstring); kept as a sanity check on top of the above.
    assert re.search(r"202[5-9]", text), f"no recent year in response: {text[:300]!r}"
