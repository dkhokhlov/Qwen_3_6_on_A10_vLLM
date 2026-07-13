#!/usr/bin/env python3
"""LiteLLM proxy plugin: wake vLLM on request + idle-stop it.

Wiring
  env LITELLM_WORKER_STARTUP_HOOKS=litellm_callbacks:start_background_tasks
  runs inside proxy_startup_event (the Uvicorn lifespan). It:
    1. boots the idle watcher + a startup health bootstrap, and
    2. registers a CustomLogger that wakes the backend before any completion.

Why a registered CustomLogger with TWO hooks, not litellm_settings.callbacks:
  - LiteLLM's config-string callback resolver only matches BUILT-IN integration
    names (langfuse/otel/...); an arbitrary `module.handler` string is left as a
    bare string and NEVER imported, so the hook silently never fires. The proxy
    has no auto-import path for a custom callback module. Fix: append the INSTANCE
    to litellm.callbacks from the startup hook (after LiteLLM has finished loading).
  - The two completion paths dispatch DIFFERENT pre-call hooks, so Handler
    overrides both:
      * router        /v1/chat/completions -> async_pre_call_hook
        (dispatched from ProxyLogging via _callback_capabilities, which reads
        litellm.callbacks live and recomputes on membership change)
      * pass-through  /v1/messages         -> async_pre_request_hook
        (anthropic_messages -> _execute_pre_request_hooks iterates
        litellm.callbacks directly, before the upstream call)
  Both hooks just call backend.ensure_up(); returning None leaves the request
  unmodified. Waking here fires ONLY on real LLM calls, so /v1/models, /health,
  /metrics (served cold from config) never wake vLLM.

Container control talks to the Engine API via httpx (the litellm image ships no
docker SDK). Under compose it reaches the docker-sock-proxy sidecar over HTTP
(DOCKER_API_BASE), which holds the host socket and is whitelisted to
start/stop + `/_ping`; with DOCKER_API_BASE unset it falls back to
the mounted /var/run/docker.sock UDS (local dev). The startup-hook loader
uses plain importlib.import_module, so the compose service sets PYTHONPATH=/app.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import httpx
import litellm
from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("litellm_callbacks")
# Force the lifecycle logger to emit at INFO regardless of the root level, so
# wake/stop events show up in `make litellm-logs` (LiteLLM otherwise leaves this
# child logger below WARNING and the events are invisible).
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

BACKEND_CONTAINER = os.environ.get("BACKEND_CONTAINER", "vllm-qwen")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://vllm:8000").rstrip("/")
IDLE_SECONDS = float(os.environ.get("IDLE_SECONDS", "900"))
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "10"))
WAKE_TIMEOUT_SECONDS = float(os.environ.get("WAKE_TIMEOUT_SECONDS", "240"))
# A compose cold boot can exceed a single wake window; give bootstrap a long leash.
BOOT_TIMEOUT_SECONDS = float(os.environ.get("BOOT_TIMEOUT_SECONDS", "600"))
# Compose: base URL of the docker-sock-proxy sidecar (whitelists start/stop + /_ping).
# Repo-specific name (not DOCKER_HOST -- that implies tcp:// / unix:// to Docker
# tooling; this client wants a plain http:// URL). Unset -> UDS fallback in Backend.
DOCKER_API_BASE = os.environ.get("DOCKER_API_BASE")
DOCKER_SOCK = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")

# Server-side max_tokens cap. Claude Code's gateway discovery ignores token limits,
# so it sends its built-in max_tokens=32000 on EVERY path -- including subagents and
# the small/fast model, which ignore the client-side CLAUDE_CODE_MAX_OUTPUT_TOKENS
# env var (claude-code issue #25569). The deployment hook below is the single point
# that sees OpenAI-shape kwargs on BOTH endpoints after Anthropic->OpenAI conversion,
# so clamping here covers every request regardless of what the client sent. Per-stack
# via compose env (moe=16384, dense=8192); must satisfy PCT*WINDOW + cap <= WINDOW.
MAX_TOKENS_CAP = int(os.environ.get("CLAUDE_QWEN_MAX_TOKENS_CAP", "16384"))

# Inject the real prompt token count into the streamed message_start.usage.input_tokens
# so Claude Code's auto-compact tracker grows and fires. The tracker reads the STREAMED
# message_start usage (NOT the terminal result usage); LiteLLM hardcodes that field to 0
# (the real count arrives only in the terminal message_delta), so on the Qwen stack the
# tracker stays at 0 and compaction NEVER fires -- context grows unbounded to a 400
# (confirmed: an in-process probe reached result.usage.input_tokens=115152 at
# WINDOW=40000/trigger=32000 with zero compact_boundary events). The ~/bin/glm wrapper
# works because z.ai reports message_start usage correctly (fires ~70%). Fix: preflight
# vLLM POST /tokenize (chat-template-applied token count) and inject that count into the
# first message_start chunk. Gated to the /v1/messages streaming path; no-op everywhere
# else. See _preflight_input_tokens + async_post_call_streaming_iterator_hook.
#
# ACCURACY CAVEAT (unresolved as of this checkpoint): /tokenize matches vLLM's real
# prompt_tokens EXACTLY for messages-only and for tools sent DIRECT to vLLM, but
# UNDERCOUNTS on the /v1/messages path when the request carries tools: LiteLLM injects a
# tool-rendering system prompt (proportional to the tools schema, ~20% of its token count)
# AFTER the deployment hook, so the hook sees pre-injection kwargs. For real claude-code
# (~28k-token fixed tools schema) the offset is ~5800 tokens -- large enough to push the
# dense stack past its safety margin (trigger 51200, overflow >55808). The no-tools path
# is exact and compaction fires correctly there; the tools path fires LATE. Needs the
# injection replicated in the preflight, a later hook, or a pivot to LiteLLM
# context_management. See plan + git history for the decisive direct-vs-proxy test.
INJECT_STREAMED_USAGE = os.environ.get(
    "CLAUDE_QWEN_INJECT_STREAMED_USAGE", "1"
).strip().lower() in ("1", "true", "yes", "on")
# litellm_call_id -> preflight input-token count. Set in async_pre_call_deployment_hook,
# consumed+popped in async_post_call_streaming_iterator_hook. litellm_call_id is the same
# value at both hooks (self.data flows to kwargs and to request_data -- see plan).
_USAGE_INJECT: dict[str, int] = {}


def _metric(line: str) -> int:
    # Prometheus exposition line: "name{labels} value"
    return int(float(line.split()[-1]))


def _extract_reasoning(message: dict) -> str | None:
    """Pull prior-turn reasoning out of an assistant message, Anthropic or OpenAI shape.

    The /v1/messages adapter attaches Anthropic `thinking_blocks` -- a list of
    {"type":"thinking","thinking":"..."}; the OpenAI path uses a top-level
    `reasoning` string. vLLM's qwen3 chat_template renders PRIOR reasoning only
    from `reasoning_content`, so either must be normalized to that key.
    """
    parts: list[str] = []
    blocks = message.get("thinking_blocks")
    if isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, dict):
                t = b.get("thinking") or b.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
    if not parts:
        r = message.get("reasoning")
        if isinstance(r, str) and r:
            parts.append(r)
    return "\n".join(parts) if parts else None


def _preserve_requested(kwargs: dict) -> str | None:
    """The alias (or flag) marking this a `-preserve` request, else None.

    `kwargs["model"]` at the deployment hook is usually the client-facing alias,
    but it can land as the deployment model (hosted_vllm/...). Probe model group
    across the places litellm stashes it, then fall back to the actual
    chat_template_kwarg, so the gate never silently no-ops on the wrong field.
    """
    for key in ("model", "model_group"):
        v = kwargs.get(key)
        if isinstance(v, str) and v.endswith("-preserve"):
            return v
    md = kwargs.get("metadata")
    v = md.get("model_group") if isinstance(md, dict) else None
    if isinstance(v, str) and v.endswith("-preserve"):
        return v
    eb = kwargs.get("extra_body")
    ctk = eb.get("chat_template_kwargs") if isinstance(eb, dict) else None
    if isinstance(ctk, dict) and ctk.get("preserve_thinking") is True:
        return "<preserve_thinking=true>"
    return None


def _clamp_max_tokens(kwargs: dict) -> bool:
    """Cap max_tokens/max_completion_tokens to MAX_TOKENS_CAP in place.

    Claude Code sends its built-in max_tokens=32000 (gateway discovery ignores the
    advertised limit) on every path -- including subagents and the small/fast model,
    which ignore CLAUDE_CODE_MAX_OUTPUT_TOKENS. The deployment hook is the one point
    that sees OpenAI-shape kwargs on both endpoints, so this is the deterministic
    backstop. Clamp only when the client asked for more than the cap; leave absent
    fields to vLLM's default. Returns whether anything was changed, so the hook can
    return kwargs (apply) only when a change occurred (LiteLLM skips on None).
    """
    changed = False
    for key in ("max_tokens", "max_completion_tokens"):
        v = kwargs.get(key)
        if isinstance(v, (int, float)) and v > MAX_TOKENS_CAP:
            log.info("max_tokens clamp: %s %d -> %d", key, int(v), MAX_TOKENS_CAP)
            kwargs[key] = MAX_TOKENS_CAP
            changed = True
    return changed


def _resolve_served_model(kwargs: dict) -> str | None:
    """vLLM /tokenize needs the served-model-name (qwen3.6-35b-a3b), NOT the litellm
    deployment id (hosted_vllm/qwen3.6-35b-a3b) nor a client alias (-preserve/-nothink,
    which vLLM does not register -> 404). kwargs["model"] at the deployment hook is
    usually the alias (no '/') but can be the deployment id; the litellm deployment env
    is the single vLLM model for this stack. Strip the provider prefix from whichever
    carries it. Returns None only if neither source is set (preflight skipped)."""
    m = kwargs.get("model")
    if isinstance(m, str) and "/" in m:
        return m.split("/", 1)[1]
    mm = os.environ.get("LITELLM_VLLM_MODEL", "").strip()
    if "/" in mm:
        return mm.split("/", 1)[1]
    return mm or None


def _chat_template_kwargs(kwargs: dict) -> dict | None:
    """The chat_template_kwargs the chosen alias sends to vLLM (enable_thinking /
    preserve_thinking). These change the chat-template render and thus the token
    count, so /tokenize must receive the same kwargs the generation uses (verified:
    nothink=18 vs thinking-on=16 for one message)."""
    eb = kwargs.get("extra_body")
    ctk = eb.get("chat_template_kwargs") if isinstance(eb, dict) else None
    return ctk if isinstance(ctk, dict) else None


async def _preflight_input_tokens(
    *, client, model: str, messages: list, tools=None, chat_template_kwargs=None
) -> int | None:
    """Chat-template-applied prompt token count via vLLM POST /tokenize. Equals vLLM's
    upcoming prompt_tokens EXACTLY when tokenizing the SAME payload vLLM receives
    (verified live direct-to-vLLM, incl. a ~19k-char tools schema: 4997==4997). On the
    /v1/messages path LiteLLM adds a tool-rendering system prompt AFTER the deployment
    hook, so tokenizing the hook's pre-injection kwargs UNDERCOUNTS by ~20% of the tools
    schema -- see the ACCURACY CAVEAT above. Returns None on any failure so the caller
    skips injection -- never blocks the request."""
    payload: dict = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    try:
        r = await client.post(f"{BACKEND_URL}/tokenize", json=payload, timeout=10.0)
        if r.status_code == 200:
            return int(r.json().get("count") or 0)
        log.warning("preflight /tokenize http=%d body=%s", r.status_code, r.text[:200])
    except Exception as exc:  # vLLM down, /tokenize unsupported, network -- skip injection
        log.warning("preflight /tokenize failed: %s", exc)
    return None


async def _stash_preflight_usage(kwargs: dict, call_type) -> None:
    """Preflight-count the prompt and stash it by litellm_call_id for the streaming
    iterator hook to inject. Gated to streamed /v1/messages (the anthropic path): the
    OpenAI path's chunks aren't message_start dicts, and non-streaming /v1/messages
    already returns correct terminal usage (no iterator hook to consume a stash -> it
    would leak). call_type is a CallTypes enum (str, Enum) or None; == "anthropic_messages"
    covers both the enum and a bare string. Does not modify kwargs."""
    if not INJECT_STREAMED_USAGE:
        return
    if call_type != "anthropic_messages":
        return
    if not kwargs.get("stream"):
        return
    model = _resolve_served_model(kwargs)
    if not model:
        return
    count = await _preflight_input_tokens(
        client=backend._client,
        model=model,
        messages=kwargs.get("messages") or [],
        tools=kwargs.get("tools") or None,
        chat_template_kwargs=_chat_template_kwargs(kwargs),
    )
    if not count:
        return
    cid = kwargs.get("litellm_call_id")
    if not cid:
        return
    _USAGE_INJECT[cid] = count
    log.info("preflight input_tokens=%d model=%s call_id=%s", count, model, str(cid)[:8])


def _inject_usage_into_message_start(chunk, count: int):
    """Rewrite usage.input_tokens=count in the first message_start chunk and return the
    rewritten chunk; return None if the chunk is not a message_start (or unparseable),
    so the caller yields the original unchanged and keeps scanning. The chunk is a
    bytes/str SSE frame on the /v1/messages path (the anthropic pass-through serializes
    to SSE BEFORE this hook -- the frame is `event: message_start\\ndata: {json}\\n\\n`);
    a dict path is retained as a defensive fallback. Mirrors LiteLLM's own
    `_inject_cost_into_sse_frame_str` SSE-rewrite pattern. Only the caller's first
    message_start is rewritten (tracked by the `injected` flag)."""
    # Dict path (defensive -- not seen on the current anthropic stream, which is bytes).
    if isinstance(chunk, dict):
        if chunk.get("type") == "message_start" and isinstance(chunk.get("message"), dict):
            usage = chunk["message"].get("usage")
            if not isinstance(usage, dict):
                usage = {}
                chunk["message"]["usage"] = usage
            usage["input_tokens"] = count
            return chunk
        return None
    # bytes/str SSE-frame path.
    if isinstance(chunk, (bytes, bytearray)):
        text = bytes(chunk).decode("utf-8", errors="replace")
        was_bytes = True
    elif isinstance(chunk, str):
        text = chunk
        was_bytes = False
    else:
        return None
    if '"message_start"' not in text:
        return None
    lines = text.split("\n")
    data_idx = next((i for i, ln in enumerate(lines) if ln.startswith("data: ")), None)
    if data_idx is None:
        return None
    try:
        obj = json.loads(lines[data_idx][6:])
    except (ValueError, TypeError):
        return None
    if obj.get("type") != "message_start":
        return None
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    usage = msg.get("usage")
    if not isinstance(usage, dict):
        usage = {}
        msg["usage"] = usage
    usage["input_tokens"] = count
    # Match LiteLLM's separators (", " / ": ") so the frame shape is stable; clients
    # parse the JSON, so the whitespace is not semantically load-bearing.
    lines[data_idx] = "data: " + json.dumps(obj, separators=(", ", ": "))
    out = "\n".join(lines)
    return out.encode("utf-8") if was_bytes else out


class Backend:
    """vLLM lifecycle: start on demand, stop after sustained inference-idle."""

    def __init__(self) -> None:
        self._up = False
        self.idle_since: float | None = None
        self._wake_lock = asyncio.Lock()
        # Health + metrics client over the compose network.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        # Engine API: over the docker-sock-proxy sidecar (HTTP) under compose, or the
        # host unix socket (UDS) for local dev. Both POST start/stop calls below are
        # relative, so they resolve against either base_url unchanged.
        if DOCKER_API_BASE:
            self._docker = httpx.AsyncClient(
                base_url=DOCKER_API_BASE, timeout=httpx.Timeout(60.0, connect=10.0))
        else:
            transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCK)
            self._docker = httpx.AsyncClient(
                transport=transport, base_url="http://docker",
                timeout=httpx.Timeout(60.0, connect=10.0))

    # -- lifecycle --------------------------------------------------------
    async def bootstrap(self) -> None:
        """At proxy startup, mark UP once the warm-started vLLM reports healthy.

        Without this a freshly-started but UNUSED stack never reaches _up, so the
        idle watcher could never stop it -- idle power-down would silently never fire.
        A failure here leaves _up False; the first request wakes it (ensure_up).
        """
        try:
            await self._wait_health(BOOT_TIMEOUT_SECONDS)
            self._up = True
            self.idle_since = None
            log.info("backend healthy at startup; idle-stop armed (idle=%.0fs)", IDLE_SECONDS)
        except TimeoutError:
            self._up = False
            log.warning("backend not healthy within %.0fs; will wake on first request",
                        BOOT_TIMEOUT_SECONDS)

    async def _probe(self) -> bool:
        """Fast liveness check; False if vLLM is down or still booting.

        `_up` is a cache that can go stale when vLLM is stopped by anything
        OTHER than idle_watch._stop() -- an external `docker stop`, a crash
        under restart:"no", or a host-driven clean shutdown. Probing before
        trusting `_up` makes a request self-heal instead of calling into a dead
        backend (the failure mode where ensure_up short-circuits on a stale
        `_up=True` and the upstream call dies with "Cannot connect to vllm:8000").
        """
        try:
            return (await self._client.get(f"{BACKEND_URL}/health", timeout=2.0)).status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_up(self, force: bool = False) -> None:
        """Start vLLM if it is down; coalesce concurrent callers behind a lock.

        On the hot path (already up) we still reset idle_since: a request is now
        in flight, so the idle timer must be pushed out before vLLM registers
        the request in /metrics -- otherwise idle_watch could stop the backend
        in the handoff window (a race at the IDLE_SECONDS boundary).
        """
        if not force and self._up and await self._probe():
            self.idle_since = None
            return
        async with self._wake_lock:
            if not force and self._up and await self._probe():
                self.idle_since = None
                return
            log.info("waking backend %s", BACKEND_CONTAINER)
            await self._start_if_needed()
            await self._wait_health(WAKE_TIMEOUT_SECONDS)
            self._up = True
            self.idle_since = None
            log.info("backend up; idle timer reset")

    async def _start_if_needed(self) -> None:
        # Engine API: POST /containers/{id}/start -> 204 ok, 304 already running,
        # 404 unknown container. 304 is the common (already-up) case; treat as ok.
        r = await self._docker.post(f"/containers/{BACKEND_CONTAINER}/start")
        if r.status_code == 404:
            raise RuntimeError(f"backend container {BACKEND_CONTAINER!r} not found")
        if r.status_code not in (204, 304):
            r.raise_for_status()

    async def _stop(self) -> None:
        try:
            r = await self._docker.post(
                f"/containers/{BACKEND_CONTAINER}/stop", params={"t": 10})
            if r.status_code not in (204, 304):
                r.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("stop failed: %s", exc)
        self._up = False
        self.idle_since = None
        log.info("backend stopped (idled %.0fs)", IDLE_SECONDS)

    async def _wait_health(self, timeout_s: float) -> None:
        url = f"{BACKEND_URL}/health"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if (await self._client.get(url, timeout=5.0)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3.0)
        raise TimeoutError(f"backend not healthy within {timeout_s:.0f}s")

    # -- idle watch -------------------------------------------------------
    async def _backend_idle(self) -> bool | None:
        """True if no inference is in flight; None if /metrics is unreadable."""
        try:
            r = await self._client.get(f"{BACKEND_URL}/metrics", timeout=5.0)
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        running = waiting = None
        swapped = 0  # optional in vLLM v1; treat absent as 0
        for line in r.text.splitlines():
            if line.startswith("vllm:num_requests_running{"):
                running = _metric(line)
            elif line.startswith("vllm:num_requests_waiting{"):
                waiting = _metric(line)
            elif line.startswith("vllm:num_requests_swapped{"):
                swapped = _metric(line)
        if running is None or waiting is None:
            return None
        return running == 0 and waiting == 0 and swapped == 0

    async def idle_watch(self) -> None:
        while True:
            await asyncio.sleep(POLL_SECONDS)
            try:
                if not self._up:
                    continue
                idle = await self._backend_idle()
                if idle is None:
                    continue  # metrics unreadable (e.g. already stopped); leave state
                now = time.monotonic()
                if idle:
                    if self.idle_since is None:
                        self.idle_since = now
                    elif now - self.idle_since >= IDLE_SECONDS:
                        await self._stop()
                else:
                    self.idle_since = None
            except Exception as exc:  # never let the watcher die
                log.exception("idle_watch iteration failed: %s", exc)


backend = Backend()

# Strong-ref so the asyncio scheduler doesn't GC the background tasks.
_tasks: set[asyncio.Task] = set()


class Handler(CustomLogger):
    """Wake the backend before a completion (both proxy paths) and, for `-preserve`
    models, re-attach prior-turn reasoning the adapter would otherwise drop.

    The two wake hooks exist because the router and the Anthropic pass-through
    dispatch different pre-call hooks (see module docstring); each just calls
    backend.ensure_up() (a no-op once up) and returns None.

    The deployment hook fires inside wrapper_async AFTER the adapter produced
    OpenAI-shape kwargs (utils.py:1606) -- the only point where a reasoning_content
    set here reaches vLLM: on /v1/messages, async_pre_call_hook sees Anthropic-shape
    data and the adapter rebuilds prior assistant turns with thinking_blocks (never
    reasoning_content) AFTER it, so the qwen3 template's prior-reasoning render
    would be empty without this. See the method below.
    """

    # Router path: /v1/chat/completions (+ /v1/completions, /v1/embeddings...).
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        await backend.ensure_up()
        return None

    # Pass-through path: /v1/messages (anthropic_messages -> _execute_pre_request_hooks).
    async def async_pre_request_hook(self, model, messages, kwargs):
        await backend.ensure_up()
        return None

    # Both endpoints, after Anthropic->OpenAI conversion. The qwen3 template renders
    # PRIOR reasoning only from `reasoning_content`; the adapter attaches Anthropic
    # thinking_blocks instead, so a -preserve request would lose every prior <think>.
    # Normalize thinking_blocks/reasoning -> reasoning_content on prior assistant
    # turns. Return contract (utils.py:1139): return kwargs to apply it, None to skip.
    # The INFO line is also the verify-probe: counts reveal whether the client strips
    # older turns' thinking (n_tb=0 -> cache needed) vs mapping alone suffices.
    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        await _stash_preflight_usage(kwargs, call_type)
        clamped = _clamp_max_tokens(kwargs)
        alias = _preserve_requested(kwargs)
        if alias is None:
            # Non-preserve: only the clamp may have changed kwargs; return it so LiteLLM
            # applies the cap (returning None would skip even an in-place clamp).
            return kwargs if clamped else None
        messages = kwargs.get("messages") or []
        rebuilt: list = []
        n_tb = n_rs = n_rc = n_set = 0
        for m in messages:
            if not (isinstance(m, dict) and m.get("role") == "assistant"):
                rebuilt.append(m)
                continue
            if m.get("reasoning_content"):
                n_rc += 1
                rebuilt.append(m)
                continue
            if isinstance(m.get("thinking_blocks"), list) and m["thinking_blocks"]:
                n_tb += 1
            if isinstance(m.get("reasoning"), str) and m["reasoning"]:
                n_rs += 1
            rc = _extract_reasoning(m)
            if rc:
                nm = dict(m)
                nm["reasoning_content"] = rc
                rebuilt.append(nm)
                n_set += 1
            else:
                rebuilt.append(m)
        log.info(
            "preserve hook: alias=%s msgs=%d prior-assistant thinking_blocks=%d "
            "reasoning=%d had_rc=%d set_rc=%d",
            alias, len(messages), n_tb, n_rs, n_rc, n_set,
        )
        if n_set:
            kwargs["messages"] = rebuilt
        # Apply if either the clamp or the preserve mapping changed anything.
        return kwargs if (n_set or clamped) else None

    # Streaming iterator hook (both endpoints; fires only for streaming responses).
    # LiteLLM dispatches per-callback overrides here (kind=="override"). On the
    # /v1/messages path the anthropic pass-through serializes to an SSE frame BEFORE
    # this hook, so the chunk here is BYTES (`event: message_start\ndata: {json}\n\n`),
    # not a dict -- _inject_usage_into_message_start parses that SSE frame and rewrites
    # the FIRST message_start's usage.input_tokens to the preflight count stashed by
    # _stash_preflight_usage (popped by litellm_call_id). A dict path is retained as a
    # defensive fallback. No stash -> pass through unchanged (the /v1/chat/completions
    # path, a failed preflight, or non-streaming).
    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response, request_data
    ):
        cid = request_data.get("litellm_call_id") if isinstance(request_data, dict) else None
        count = _USAGE_INJECT.pop(cid, None) if cid else None
        if count is None:
            async for chunk in response:
                yield chunk
            return
        injected = False
        async for chunk in response:
            if not injected:
                rewritten = _inject_usage_into_message_start(chunk, count)
                if rewritten is not None:
                    chunk = rewritten
                    injected = True
                    log.info("injected input_tokens=%d into message_start call_id=%s",
                             count, str(cid)[:8])
            yield chunk


async def start_background_tasks() -> None:
    """LITELLM_WORKER_STARTUP_HOOKS entrypoint (run inside the lifespan loop)."""
    for coro in (backend.bootstrap(), backend.idle_watch()):
        t = asyncio.create_task(coro)
        _tasks.add(t)
        t.add_done_callback(_tasks.discard)
    # Append the wake hook to litellm.callbacks (NOT the config string -- the
    # resolver never imports custom modules). Both dispatch sites read this list:
    # the router via _callback_capabilities (recomputed on membership change),
    # the pass-through directly. Idempotent in case the hook ever re-runs.
    if not any(isinstance(c, Handler) for c in litellm.callbacks):
        litellm.callbacks.append(Handler())
        log.info("wake handler registered on litellm.callbacks")
