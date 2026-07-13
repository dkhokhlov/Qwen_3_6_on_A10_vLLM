"""Unit tests for the pure (no-I/O) helpers and the preserve-thinking deployment hook
in litellm_callbacks.py. No mocks needed: these are functions over plain dicts."""
import pytest

import litellm_callbacks as L


# --------------------------------------------------------------------------- #
# _metric: parse the trailing value of a Prometheus exposition line
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("line, expected", [
    ('vllm:num_requests_running{x="1"} 5', 5),
    ("vllm:foo 0", 0),
    ("vllm:foo 3.0", 3),     # float -> int truncation
    ("vllm:foo 12.7", 12),
])
def test_metric(line, expected):
    assert L._metric(line) == expected


# --------------------------------------------------------------------------- #
# _extract_reasoning: normalize prior-turn reasoning from Anthropic/OpenAI shapes
# --------------------------------------------------------------------------- #
def test_extract_reasoning_thinking_blocks():
    msg = {"thinking_blocks": [{"type": "thinking", "thinking": "deliberation"}]}
    assert L._extract_reasoning(msg) == "deliberation"


def test_extract_reasoning_text_key():
    msg = {"thinking_blocks": [{"text": "via text"}]}
    assert L._extract_reasoning(msg) == "via text"


def test_extract_reasoning_multiple_blocks_joined():
    msg = {"thinking_blocks": [{"thinking": "a"}, {"thinking": "b"}]}
    assert L._extract_reasoning(msg) == "a\nb"


def test_extract_reasoning_openai_reasoning_fallback():
    assert L._extract_reasoning({"reasoning": "openai-style"}) == "openai-style"


def test_extract_reasoning_thinking_takes_precedence_over_reasoning():
    msg = {"thinking_blocks": [{"thinking": "blocks"}], "reasoning": "ignored"}
    assert L._extract_reasoning(msg) == "blocks"


def test_extract_reasoning_empty_returns_none():
    assert L._extract_reasoning({}) is None
    assert L._extract_reasoning({"thinking_blocks": []}) is None
    assert L._extract_reasoning({"reasoning": ""}) is None


# --------------------------------------------------------------------------- #
# _preserve_requested: detect the -preserve alias/flag across stash locations
# --------------------------------------------------------------------------- #
def test_preserve_requested_model():
    assert L._preserve_requested({"model": "qwen3.6-27b-preserve"}) == "qwen3.6-27b-preserve"


def test_preserve_requested_model_group():
    assert L._preserve_requested({"model_group": "x-preserve"}) == "x-preserve"


def test_preserve_requested_metadata_model_group():
    assert L._preserve_requested({"metadata": {"model_group": "y-preserve"}}) == "y-preserve"


def test_preserve_requested_chat_template_kwarg():
    kw = {"extra_body": {"chat_template_kwargs": {"preserve_thinking": True}}}
    assert L._preserve_requested(kw) == "<preserve_thinking=true>"


def test_preserve_requested_first_match_wins():
    kw = {"model": "a-preserve", "metadata": {"model_group": "b-preserve"}}
    assert L._preserve_requested(kw) == "a-preserve"


@pytest.mark.parametrize("kw", [
    {},
    {"model": "qwen3.6-27b"},
    {"model": "qwen3.6-27b-nothink"},
    {"model_group": "qwen3.6-27b"},
    {"extra_body": {"chat_template_kwargs": {"preserve_thinking": False}}},
])
def test_preserve_requested_none(kw):
    assert L._preserve_requested(kw) is None


# --------------------------------------------------------------------------- #
# Handler.async_pre_call_deployment_hook: re-attach prior-turn reasoning for -preserve
# --------------------------------------------------------------------------- #
async def test_deployment_hook_non_preserve_returns_none():
    h = L.Handler()
    kwargs = {"model": "qwen3.6-27b", "messages": [{"role": "assistant", "content": "x"}]}
    assert await h.async_pre_call_deployment_hook(kwargs, "acompletion") is None


async def test_deployment_hook_sets_reasoning_content_from_thinking_blocks():
    h = L.Handler()
    kwargs = {
        "model": "qwen3.6-27b-preserve",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok",
             "thinking_blocks": [{"type": "thinking", "thinking": "prior chain"}]},
        ],
    }
    out = await h.async_pre_call_deployment_hook(kwargs, "acompletion")
    assert out is not None
    assistant = out["messages"][1]
    assert assistant["reasoning_content"] == "prior chain"
    # non-assistant messages pass through untouched
    assert out["messages"][0] == kwargs["messages"][0]


async def test_deployment_hook_sets_reasoning_content_from_reasoning_string():
    h = L.Handler()
    kwargs = {
        "model": "qwen3.6-27b-preserve",
        "messages": [{"role": "assistant", "content": "ok", "reasoning": "openai-style"}],
    }
    out = await h.async_pre_call_deployment_hook(kwargs, "acompletion")
    assert out is not None
    assert out["messages"][0]["reasoning_content"] == "openai-style"


async def test_deployment_hook_skips_assistant_already_having_reasoning_content():
    h = L.Handler()
    kwargs = {
        "model": "qwen3.6-27b-preserve",
        "messages": [{"role": "assistant", "content": "ok", "reasoning_content": "existing"}],
    }
    # nothing to set -> returns None, leaves messages unchanged
    assert await h.async_pre_call_deployment_hook(kwargs, "acompletion") is None


async def test_deployment_hook_preserve_but_no_reasoning_returns_none():
    h = L.Handler()
    kwargs = {
        "model": "qwen3.6-27b-preserve",
        "messages": [{"role": "assistant", "content": "ok"}],  # no thinking/reasoning
    }
    assert await h.async_pre_call_deployment_hook(kwargs, "acompletion") is None


# --------------------------------------------------------------------------- #
# _clamp_max_tokens + deployment hook: server-side cap on Claude Code's 32000
# default (subagents/small-fast ignore CLAUDE_CODE_MAX_OUTPUT_TOKENS; this is the
# deterministic backstop covering every request path on both endpoints).
# --------------------------------------------------------------------------- #
def test_clamp_reduces_oversized_max_tokens():
    kw = {"max_tokens": 32000}
    assert L._clamp_max_tokens(kw) is True
    assert kw["max_tokens"] == L.MAX_TOKENS_CAP


def test_clamp_leaves_undersized_max_tokens_unchanged():
    kw = {"max_tokens": 4096}
    assert L._clamp_max_tokens(kw) is False
    assert kw["max_tokens"] == 4096


def test_clamp_handles_max_completion_tokens_field():
    kw = {"max_completion_tokens": 32000}
    assert L._clamp_max_tokens(kw) is True
    assert kw["max_completion_tokens"] == L.MAX_TOKENS_CAP


def test_clamp_noop_when_no_max_tokens_field():
    kw = {"messages": []}
    assert L._clamp_max_tokens(kw) is False
    assert kw == {"messages": []}


async def test_deployment_hook_clamps_non_preserve_and_returns_kwargs():
    # The 400 path: a non-preserve (strip/nothink) request sends max_tokens=32000. The
    # hook must clamp it AND return kwargs so LiteLLM applies the cap (None would skip).
    h = L.Handler()
    kwargs = {"model": "qwen3.6-35b-a3b-nothink", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 32000}
    out = await h.async_pre_call_deployment_hook(kwargs, "acompletion")
    assert out is not None
    assert out["max_tokens"] == L.MAX_TOKENS_CAP


async def test_deployment_hook_non_preserve_no_max_tokens_returns_none():
    # No max_tokens -> clamp no-op -> non-preserve -> None (unchanged behavior).
    h = L.Handler()
    kwargs = {"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "hi"}]}
    assert await h.async_pre_call_deployment_hook(kwargs, "acompletion") is None


async def test_deployment_hook_preserve_clamps_even_when_no_reasoning():
    # A -preserve request with max_tokens=32000 but no prior reasoning: nothing to map,
    # but the clamp still applies -> returns kwargs (not None).
    h = L.Handler()
    kwargs = {"model": "qwen3.6-27b-preserve",
              "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32000}
    out = await h.async_pre_call_deployment_hook(kwargs, "acompletion")
    assert out is not None
    assert out["max_tokens"] == L.MAX_TOKENS_CAP


# --------------------------------------------------------------------------- #
# Streamed-usage injection: preflight /tokenize + message_start rewrite.
# Auto-compact reads the STREAMED message_start.usage.input_tokens (not the
# terminal result); LiteLLM hardcodes it to 0, so the tracker never grows and
# compaction never fires. Fix: preflight /tokenize at log_pre_api_call (the
# post-conversion wire body) and inject into the first message_start SSE frame.
# --------------------------------------------------------------------------- #
def test_served_model_strips_provider_prefix():
    assert L._served_model("hosted_vllm/qwen3.6-35b-a3b") == "qwen3.6-35b-a3b"


def test_served_model_passes_through_bare_name():
    assert L._served_model("qwen3.6-35b-a3b") == "qwen3.6-35b-a3b"


def test_served_model_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(L.os, "environ", {"LITELLM_VLLM_MODEL": "hosted_vllm/qwen3.6-27b"})
    assert L._served_model(None) == "qwen3.6-27b"


class _FakeResp:
    def __init__(self, data): self._data = data
    def read(self): return self._data
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_preflight_sync_returns_count(monkeypatch):
    monkeypatch.setattr(L.urllib.request, "urlopen",
                        lambda req, timeout=10.0: _FakeResp(b'{"count": 538}'))
    assert L._preflight_input_tokens_sync(
        model="qwen3.6-35b-a3b", messages=[{"role": "user", "content": "hi"}]) == 538


def test_preflight_sync_returns_none_on_error(monkeypatch):
    monkeypatch.setattr(L.urllib.request, "urlopen", lambda req, timeout=10.0: (_ for _ in ()).throw(OSError("down")))
    assert L._preflight_input_tokens_sync(model="qwen3.6-35b-a3b", messages=[]) is None


def test_inject_usage_rewrites_bytes_message_start():
    chunk = (b'event: message_start\ndata: '
             b'{"type":"message_start","message":{"usage":{"input_tokens":0}}}\n\n')
    out = L._inject_usage_into_message_start(chunk, 538)
    assert out is not None
    assert b'"input_tokens": 538' in out
    assert out.startswith(b'event: message_start')


def test_inject_usage_returns_none_for_non_message_start_bytes():
    chunk = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    assert L._inject_usage_into_message_start(chunk, 538) is None


def test_inject_usage_returns_none_for_unparseable_data():
    chunk = b'event: message_start\ndata: {not json}\n\n'
    assert L._inject_usage_into_message_start(chunk, 538) is None


def test_inject_usage_rewrites_dict_message_start():
    chunk = {"type": "message_start", "message": {"usage": {"input_tokens": 0}}}
    out = L._inject_usage_into_message_start(chunk, 15)
    assert out["message"]["usage"]["input_tokens"] == 15


def test_inject_usage_returns_none_for_non_dict_non_bytes():
    assert L._inject_usage_into_message_start(123, 538) is None


def test_log_pre_api_call_stashes_count(monkeypatch):
    monkeypatch.setattr(L, "_preflight_input_tokens_sync", lambda **kw: 538)
    h = L.Handler()
    cid = "call-abc"
    kwargs = {"call_type": "anthropic_messages", "stream": True, "litellm_call_id": cid,
              "additional_args": {"complete_input_dict": {
                  "model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": "hi"}],
                  "tools": [], "chat_template_kwargs": {"preserve_thinking": False}}}}
    try:
        h.log_pre_api_call("qwen3.6-35b-a3b", [], kwargs)
        assert L._USAGE_INJECT.get(cid) == 538
    finally:
        L._USAGE_INJECT.pop(cid, None)


def test_log_pre_api_call_skips_wrong_call_type(monkeypatch):
    monkeypatch.setattr(L, "_preflight_input_tokens_sync", lambda **kw: 999)
    h = L.Handler()
    kwargs = {"call_type": "acompletion", "stream": True, "litellm_call_id": "x",
              "additional_args": {"complete_input_dict": {"model": "qwen3.6-35b-a3b",
                "messages": [{"role": "user", "content": "hi"}]}}}
    h.log_pre_api_call("qwen3.6-35b-a3b", [], kwargs)
    assert "x" not in L._USAGE_INJECT


def test_log_pre_api_call_skips_non_stream(monkeypatch):
    monkeypatch.setattr(L, "_preflight_input_tokens_sync", lambda **kw: 999)
    h = L.Handler()
    kwargs = {"call_type": "anthropic_messages", "stream": False, "litellm_call_id": "x",
              "additional_args": {"complete_input_dict": {"model": "qwen3.6-35b-a3b",
                "messages": [{"role": "user", "content": "hi"}]}}}
    h.log_pre_api_call("qwen3.6-35b-a3b", [], kwargs)
    assert "x" not in L._USAGE_INJECT


def test_log_pre_api_call_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(L, "INJECT_STREAMED_USAGE", False)
    monkeypatch.setattr(L, "_preflight_input_tokens_sync", lambda **kw: 999)
    h = L.Handler()
    kwargs = {"call_type": "anthropic_messages", "stream": True, "litellm_call_id": "x",
              "additional_args": {"complete_input_dict": {"model": "qwen3.6-35b-a3b",
                "messages": [{"role": "user", "content": "hi"}]}}}
    h.log_pre_api_call("qwen3.6-35b-a3b", [], kwargs)
    assert "x" not in L._USAGE_INJECT


def test_log_pre_api_call_skips_when_preflight_returns_none(monkeypatch):
    monkeypatch.setattr(L, "_preflight_input_tokens_sync", lambda **kw: None)
    h = L.Handler()
    kwargs = {"call_type": "anthropic_messages", "stream": True, "litellm_call_id": "x",
              "additional_args": {"complete_input_dict": {"model": "qwen3.6-35b-a3b",
                "messages": [{"role": "user", "content": "hi"}]}}}
    h.log_pre_api_call("qwen3.6-35b-a3b", [], kwargs)
    assert "x" not in L._USAGE_INJECT


async def _agen(chunks):
    for c in chunks:
        yield c


async def test_iterator_hook_injects_first_message_start_and_pops_stash():
    h = L.Handler()
    cid = "call-xyz"
    L._USAGE_INJECT[cid] = 538
    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":0}}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n',
    ]
    out = []
    async for c in h.async_post_call_streaming_iterator_hook(None, _agen(chunks), {"litellm_call_id": cid}):
        out.append(c)
    assert b'"input_tokens": 538' in out[0]
    assert out[1] == chunks[1]            # subsequent chunk passes through unchanged
    assert cid not in L._USAGE_INJECT     # stash popped


async def test_iterator_hook_no_stash_passes_through():
    h = L.Handler()
    chunk = b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":0}}}\n\n'
    out = []
    async for c in h.async_post_call_streaming_iterator_hook(None, _agen([chunk]), {"litellm_call_id": "nope"}):
        out.append(c)
    assert out[0] == chunk                # unchanged when no stash entry
