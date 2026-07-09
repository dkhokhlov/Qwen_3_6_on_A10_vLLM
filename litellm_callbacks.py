#!/usr/bin/env python3
"""LiteLLM callback owning the vLLM backend lifecycle (wake-on-request + idle-stop).

Ported from sidecar/app.py (the verified idle gate). Runs inside the LiteLLM proxy
image, which ships httpx but NOT the docker SDK, so container control talks to the
Engine API over the mounted /var/run/docker.sock instead of the `docker` package.

Two entrypoints consumed by litellm_config.yaml:
  - litellm_settings.callbacks: litellm_callbacks.handler
        async_pre_call_hook wakes vLLM before each completion.
  - env LITELLM_WORKER_STARTUP_HOOKS=litellm_callbacks.start_background_tasks
        bootstrap() marks the warm-started backend UP; idle_watch() stops it after
        IDLE_SECONDS of inference-idle.  (The startup-hook loader uses plain
        importlib.import_module, so the compose service sets PYTHONPATH=/app.)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx
from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("litellm_callbacks")

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


class Handler(CustomLogger):
    """LiteLLM CustomLogger: wake the backend before each completion."""

    def __init__(self) -> None:
        self.backend = Backend()

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # Fires only on real LLM calls (not LiteLLM's own /v1/models, /health), so
        # background pollers cannot keep the GPU awake. Wake is coalesced via a lock.
        #
        # NOTE on claude-sonnet-preserve: preserve_thinking only re-renders prior
        # reasoning that is actually present in the incoming messages. Claude Code /
        # LiteLLM may strip prior assistant thinking from history, or name the field
        # `reasoning` whereas the Qwen3.6 template reads `reasoning_content`. Verify
        # at bench whether thinking round-trips; if not, re-inject HERE for the
        # preserve alias only -- set reasoning_content on prior assistant messages
        # where absent (idempotent), then `return data` so LiteLLM updates the body.
        # Left unimplemented until the /v1/messages thinking path is confirmed.
        await self.backend.ensure_up()
        return None  # proceed unchanged


# CustomLogger instance the callback loader resolves as litellm_callbacks.handler.
handler = Handler()

# Strong-ref so the asyncio scheduler doesn't GC the background tasks.
_tasks: set[asyncio.Task] = set()


async def start_background_tasks() -> None:
    """LITELLM_WORKER_STARTUP_HOOKS entrypoint (run inside the lifespan loop)."""
    for coro in (handler.backend.bootstrap(), handler.backend.idle_watch()):
        t = asyncio.create_task(coro)
        _tasks.add(t)
        t.add_done_callback(_tasks.discard)
