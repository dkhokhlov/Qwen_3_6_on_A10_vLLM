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
