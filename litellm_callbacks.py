#!/usr/bin/env python3
"""LiteLLM proxy plugin: wake vLLM on request + idle-stop it (lifecycle ported
from sidecar/app.py, the verified idle gate).

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

Container control talks to the Engine API over the mounted /var/run/docker.sock
via httpx (the litellm image ships no docker SDK). The startup-hook loader uses
plain importlib.import_module, so the compose service sets PYTHONPATH=/app.
"""
from __future__ import annotations

import asyncio
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
DOCKER_SOCK = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")


def _metric(line: str) -> int:
    # Prometheus exposition line: "name{labels} value"
    return int(float(line.split()[-1]))


class Backend:
    """vLLM lifecycle: start on demand, stop after sustained inference-idle."""

    def __init__(self) -> None:
        self._up = False
        self.idle_since: float | None = None
        self._wake_lock = asyncio.Lock()
        # Health + metrics client over the compose network.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        # Engine API over the unix socket; the URL host is cosmetic for a UDS transport.
        transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCK)
        self._docker = httpx.AsyncClient(
            transport=transport, base_url="http://docker",
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

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

    async def ensure_up(self, force: bool = False) -> None:
        """Start vLLM if it is down; coalesce concurrent callers behind a lock."""
        if not force and self._up:
            return
        async with self._wake_lock:
            if not force and self._up:
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
    """Wake the backend before a completion, on both proxy paths.

    Two hooks because the router and the Anthropic pass-through dispatch
    different pre-call hooks (see module docstring). Each is a thin wrapper over
    backend.ensure_up(), which is a no-op once the backend is already up; both
    return None so the request passes through unmodified.
    """

    # Router path: /v1/chat/completions (+ /v1/completions, /v1/embeddings...).
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        await backend.ensure_up()
        return None

    # Pass-through path: /v1/messages (anthropic_messages -> _execute_pre_request_hooks).
    async def async_pre_request_hook(self, model, messages, kwargs):
        await backend.ensure_up()
        return None


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
