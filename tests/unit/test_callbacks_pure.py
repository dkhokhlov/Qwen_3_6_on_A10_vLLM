"""Unit tests for the pure (no-I/O) helpers and the preserve-thinking deployment hook
in litellm_callbacks.py. No mocks needed: these are functions over plain dicts."""
import json

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


def test_preflight_sends_exact_payload_with_all_fields(monkeypatch):
    # The preflight MUST tokenize the EXACT post-conversion body vLLM receives
    # (messages + tools + chat_template_kwargs from complete_input_dict) -- that is what
    # makes /tokenize's count equal vLLM's upcoming prompt_tokens (the 538==538 contract).
    # Dropping tools or chat_template_kwargs undercounts -> injected input_tokens drifts
    # below vLLM's real prompt_tokens -> claude's tracker grows too slowly -> compaction
    # fires late/never. Capture the built Request and assert the payload carries every
    # field, POSTs to /tokenize.
    sent = {}
    def fake_urlopen(req, timeout=10.0):
        sent["url"] = req.full_url
        sent["data"] = json.loads(req.data.decode())
        sent["method"] = req.get_method()
        return _FakeResp(b'{"count": 9001}')
    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    tools = [{"type": "function", "function": {"name": "f"}}]
    ctk = {"preserve_thinking": False}
    count = L._preflight_input_tokens_sync(
        model="qwen3.6-35b-a3b",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools, chat_template_kwargs=ctk)
    assert count == 9001
    assert sent["url"].endswith("/tokenize")
    assert sent["method"] == "POST"
    assert sent["data"]["model"] == "qwen3.6-35b-a3b"
    assert sent["data"]["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["data"]["tools"] == tools
    assert sent["data"]["chat_template_kwargs"] == ctk


def test_preflight_omits_optional_fields_when_absent(monkeypatch):
    # tools / chat_template_kwargs are conditional; when absent they must NOT be sent
    # (vLLM /tokenize would mis-apply defaults). Payload is exactly {model, messages}.
    sent = {}
    def fake_urlopen(req, timeout=10.0):
        sent["data"] = json.loads(req.data.decode())
        return _FakeResp(b'{"count": 1}')
    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    L._preflight_input_tokens_sync(
        model="qwen3.6-35b-a3b", messages=[{"role": "user", "content": "x"}])
    assert sent["data"] == {"model": "qwen3.6-35b-a3b",
                            "messages": [{"role": "user", "content": "x"}]}
    assert "tools" not in sent["data"]
    assert "chat_template_kwargs" not in sent["data"]


def test_inject_usage_rewrites_bytes_message_start():
    chunk = (b'event: message_start\ndata: '
             b'{"type":"message_start","message":{"usage":{"input_tokens":0}}}\n\n')
    out = L._inject_usage_into_message_start(chunk, 538)
    assert out is not None
    assert b'"input_tokens": 538' in out
    assert out.startswith(b'event: message_start')


def test_inject_usage_rewrites_message_start_coalesced_after_ping():
    # A single yielded chunk may carry multiple SSE events (ping then message_start).
    # The rewriter must scan past the ping's data line and rewrite message_start,
    # not bail on the first data line. Regression guard for the silent-miss path.
    chunk = (
        b'event: ping\ndata: {"type":"ping"}\n\n'
        b'event: message_start\ndata: '
        b'{"type":"message_start","message":{"usage":{"input_tokens":0}}}\n\n'
    )
    out = L._inject_usage_into_message_start(chunk, 538)
    assert out is not None
    assert b'"input_tokens": 538' in out
    # The ping frame is preserved verbatim; message_start is the rewritten one.
    assert b'event: ping\ndata: {"type":"ping"}\n\n' in out
    assert out.count(b'"type": "message_start"') == 1


def test_inject_usage_preserves_sse_framing():
    # The split/join must preserve the trailing \n\n that delimits SSE frames.
    chunk = (b'event: message_start\ndata: '
             b'{"type":"message_start","message":{"usage":{"input_tokens":0}}}\n\n')
    out = L._inject_usage_into_message_start(chunk, 538)
    assert out is not None and out.endswith(b"\n\n")


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


def test_log_pre_api_call_passes_tools_and_template_kwargs_to_preflight(monkeypatch):
    # The hook extracts messages + tools + chat_template_kwargs from
    # additional_args.complete_input_dict (the EXACT post-conversion body vLLM receives)
    # and passes them to _preflight_input_tokens_sync. A regression that stops passing
    # tools= (the field LiteLLM injects DURING transform_request, AFTER the earlier
    # deployment hook) would silently undercount. Capture the preflight kwargs and assert
    # tools + chat_template_kwargs flow through.
    captured = {}
    monkeypatch.setattr(L, "_preflight_input_tokens_sync",
                        lambda **kw: (captured.update(kw) or 538))
    h = L.Handler()
    kwargs = {"call_type": "anthropic_messages", "stream": True, "litellm_call_id": "c",
              "additional_args": {"complete_input_dict": {
                  "model": "qwen3.6-35b-a3b",
                  "messages": [{"role": "user", "content": "hi"}],
                  "tools": [{"type": "function", "function": {"name": "f"}}],
                  "chat_template_kwargs": {"preserve_thinking": False}}}}
    try:
        h.log_pre_api_call("qwen3.6-35b-a3b", [], kwargs)
        assert captured["model"] == "qwen3.6-35b-a3b"
        assert captured["messages"] == [{"role": "user", "content": "hi"}]
        assert captured["tools"] == [{"type": "function", "function": {"name": "f"}}]
        assert captured["chat_template_kwargs"] == {"preserve_thinking": False}
    finally:
        L._USAGE_INJECT.pop("c", None)


def test_usage_inject_stash_capped_evicts_oldest(monkeypatch):
    # Backstop for the failed-streamed-request leak: a streamed request that is stashed
    # but never popped (failure before the streaming iterator hook ran) leaves a stale
    # entry. Simulate _USAGE_INJECT_CAP such failures (never popped) + one more, and
    # confirm the dict is capped and the OLDEST entry was evicted (FIFO), while the
    # fresh in-flight entry survives. Uses a tiny cap so the test is fast.
    monkeypatch.setattr(L, "_preflight_input_tokens_sync", lambda **kw: 7)
    monkeypatch.setattr(L, "_USAGE_INJECT_CAP", 8)
    L._USAGE_INJECT.clear()
    try:
        h = L.Handler()
        oldest_cid = "cid-00"
        for i in range(9):                          # 9 = cap(8) + 1 -> one eviction
            cid = f"cid-{i:02d}"
            kwargs = {"call_type": "anthropic_messages", "stream": True, "litellm_call_id": cid,
                      "additional_args": {"complete_input_dict": {"model": "qwen3.6-35b-a3b",
                        "messages": [{"role": "user", "content": "hi"}]}}}
            h.log_pre_api_call("qwen3.6-35b-a3b", [], kwargs)
        assert len(L._USAGE_INJECT) == 8             # capped, not 9
        assert oldest_cid not in L._USAGE_INJECT     # oldest evicted (FIFO)
        assert "cid-08" in L._USAGE_INJECT           # newest (in-flight) survives
    finally:
        L._USAGE_INJECT.clear()


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


async def test_iterator_hook_warns_when_stash_set_but_no_message_start(caplog):
    # Silent-failure observability (M3): if the preflight stashed a count but the stream
    # never yields a message_start frame (chunk-shape drift / stream error / empty stream),
    # the count is dropped and auto-compact never fires -- the exact regression the
    # injection exists to prevent. The hook must emit a WARNING naming the dropped count
    # so the failure is grep-able, not silent.
    import logging
    h = L.Handler()
    cid = "call-warn"
    L._USAGE_INJECT[cid] = 538
    # A stream with NO message_start -- only a content_block_delta.
    chunks = [b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n']
    out = []
    with caplog.at_level(logging.WARNING, logger=L.log.name):
        async for c in h.async_post_call_streaming_iterator_hook(None, _agen(chunks), {"litellm_call_id": cid}):
            out.append(c)
    assert cid not in L._USAGE_INJECT      # stash still popped (success or not)
    assert out == chunks                    # chunk passed through unrewritten
    assert any("NOT injected" in r.message and "538" in r.message for r in caplog.records), (
        "expected a WARNING that the stashed count was NOT injected"
    )


async def test_iterator_hook_no_warning_on_successful_injection(caplog):
    # Success path must NOT warn: a real message_start is rewritten, the info log fires,
    # and no "NOT injected" WARNING appears.
    import logging
    h = L.Handler()
    cid = "call-ok"
    L._USAGE_INJECT[cid] = 538
    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":0}}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n',
    ]
    out = []
    with caplog.at_level(logging.WARNING, logger=L.log.name):
        async for c in h.async_post_call_streaming_iterator_hook(None, _agen(chunks), {"litellm_call_id": cid}):
            out.append(c)
    assert b'"input_tokens": 538' in out[0]
    assert not any("NOT injected" in r.message for r in caplog.records), (
        "no WARNING expected on successful injection"
    )


# --------------------------------------------------------------------------- #
# Handler.async_pre_request_hook: opt-in proxy-side compaction (compact_20260112)
# injection on the /v1/messages pass-through path. Off (default) -> None, no
# mutation. On -> inject `context_management` and return kwargs (a non-None
# return REPLACES request_kwargs; the key flows to the polyfill). backend.ensure_up
# is stubbed here (the wake path is exercised live in the integration probe).
# --------------------------------------------------------------------------- #
async def _noop_ensure_up():
    return None


async def test_pre_request_hook_off_returns_none_and_does_not_mutate(monkeypatch):
    monkeypatch.setattr(L.backend, "ensure_up", _noop_ensure_up)
    monkeypatch.setattr(L, "PROXY_COMPACT", False)
    h = L.Handler()
    kwargs = {"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": "hi"}]}
    out = await h.async_pre_request_hook(kwargs["model"], kwargs["messages"], kwargs)
    assert out is None
    assert "context_management" not in kwargs   # no in-place mutation when off
    assert "drop_params" not in kwargs          # no per-request un-gate when off


async def test_pre_request_hook_on_injects_context_management(monkeypatch):
    monkeypatch.setattr(L.backend, "ensure_up", _noop_ensure_up)
    monkeypatch.setattr(L, "PROXY_COMPACT", True)
    monkeypatch.setattr(L, "PROXY_COMPACT_THRESHOLD", 90000)
    h = L.Handler()
    kwargs = {"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": "hi"}]}
    out = await h.async_pre_request_hook(kwargs["model"], kwargs["messages"], kwargs)
    assert out is kwargs                      # returns the full request_kwargs (replaces)
    edit = out["context_management"]["edits"][0]
    assert edit["type"] == "compact_20260112"
    assert edit["trigger"] == {"type": "input_tokens", "value": 90000}
    # Top-level per-request drop_params=False un-gates the polyfill for THIS opt-in
    # call, overriding the global drop_params:true the router set. Must be top-level:
    # the hook's litellm_params sub-dict is popped+discarded (messages/handler.py:261),
    # so a sub-dict key would never reach the polyfill gate.
    assert out["drop_params"] is False


async def test_pre_request_hook_threshold_env_drives_value(monkeypatch):
    monkeypatch.setattr(L.backend, "ensure_up", _noop_ensure_up)
    monkeypatch.setattr(L, "PROXY_COMPACT", True)
    monkeypatch.setattr(L, "PROXY_COMPACT_THRESHOLD", 123456)
    h = L.Handler()
    kwargs = {"model": "qwen3.6-27b", "messages": []}
    out = await h.async_pre_request_hook("qwen3.6-27b", [], kwargs)
    assert out["context_management"]["edits"][0]["trigger"]["value"] == 123456


async def test_pre_request_hook_appends_compact_to_existing_edits(monkeypatch):
    # L2: an existing client-supplied context_management must NOT be clobbered. Append the
    # compact edit to the client's edits list (client edits preserved, compact added).
    monkeypatch.setattr(L.backend, "ensure_up", _noop_ensure_up)
    monkeypatch.setattr(L, "PROXY_COMPACT", True)
    monkeypatch.setattr(L, "PROXY_COMPACT_THRESHOLD", 90000)
    h = L.Handler()
    client_edit = {"type": "clear_tool_uses"}
    kwargs = {"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": "hi"}],
              "context_management": {"edits": [client_edit]}}
    out = await h.async_pre_request_hook(kwargs["model"], kwargs["messages"], kwargs)
    edits = out["context_management"]["edits"]
    assert edits[0] is client_edit                 # client edit preserved, first
    assert edits[1]["type"] == "compact_20260112"  # compact appended
    assert len(edits) == 2
    assert out["drop_params"] is False             # polyfill still un-gated


async def test_pre_request_hook_appends_to_existing_compact_edit(monkeypatch):
    # A client that already sent a compact edit: ours appends after it (the polyfill runs
    # both in order; the second sees the already-compacted input and no-ops -- harmless).
    monkeypatch.setattr(L.backend, "ensure_up", _noop_ensure_up)
    monkeypatch.setattr(L, "PROXY_COMPACT", True)
    monkeypatch.setattr(L, "PROXY_COMPACT_THRESHOLD", 90000)
    h = L.Handler()
    client_compact = {"type": "compact_20260112",
                      "trigger": {"type": "input_tokens", "value": 50000}}
    kwargs = {"model": "qwen3.6-35b-a3b", "messages": [],
              "context_management": {"edits": [client_compact]}}
    out = await h.async_pre_request_hook(kwargs["model"], kwargs["messages"], kwargs)
    edits = out["context_management"]["edits"]
    assert len(edits) == 2
    assert edits[0]["trigger"]["value"] == 50000   # client's lower trigger preserved
    assert edits[1]["trigger"]["value"] == 90000   # ours appended


async def test_pre_request_hook_overwrites_malformed_context_management(monkeypatch):
    # Malformed existing context_management (not a dict-with-edits-list) -> can't append,
    # so create a fresh spec (overwrite). The else branch.
    monkeypatch.setattr(L.backend, "ensure_up", _noop_ensure_up)
    monkeypatch.setattr(L, "PROXY_COMPACT", True)
    monkeypatch.setattr(L, "PROXY_COMPACT_THRESHOLD", 90000)
    h = L.Handler()
    kwargs = {"model": "qwen3.6-35b-a3b", "messages": [], "context_management": "garbage"}
    out = await h.async_pre_request_hook(kwargs["model"], kwargs["messages"], kwargs)
    assert out["context_management"] == {"edits": [{
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": 90000}}]}
