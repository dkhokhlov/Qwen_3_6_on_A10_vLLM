"""Unit tests for the Backend lifecycle (wake-on-request, idle-stop) and the startup
hook registration in litellm_callbacks.py.

The module builds a `backend = Backend()` singleton at import with hard-coded httpx
clients. Because `_client`/`_docker` are plain instance attributes, we drive the
lifecycle by monkeypatching them (and time.monotonic / asyncio.sleep where a loop or
deadline is involved) — NO edits to litellm_callbacks.py. The singleton's asyncio.Lock
is recreated per test so it binds to each test's own event loop.
"""
import asyncio
import types

import httpx
import pytest

import litellm_callbacks as L


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeResp:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"fake {self.status_code}")


class FakeClient:
    """Async httpx stand-in. `responses` maps a URL substring -> FakeResp | Exception."""

    def __init__(self, responses=None):
        self._responses = responses or {}
        self.posts = []  # [(url, params)]
        self.gets = []   # [url]

    def _resolve(self, url):
        for sub, r in self._responses.items():
            if sub in url:
                if isinstance(r, BaseException):
                    raise r
                return r
        return FakeResp(404)

    async def get(self, url, timeout=None):
        self.gets.append(url)
        return self._resolve(url)

    async def post(self, url, params=None, **kwargs):
        self.posts.append((url, params))
        return self._resolve(url)


async def _atrue():
    return True


async def _afalse():
    return False


async def _anone():
    return None


@pytest.fixture(autouse=True)
def _reset_backend():
    """Reset singleton state + give each test a fresh lock bound to its own loop."""
    L.backend._up = False
    L.backend.idle_since = None
    L.backend._wake_lock = asyncio.Lock()
    yield


# --------------------------------------------------------------------------- #
# ensure_up
# --------------------------------------------------------------------------- #
async def test_ensure_up_fast_path_skips_start(monkeypatch):
    L.backend._up = True
    L.backend.idle_since = 1234.0
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/health": FakeResp(200)}))
    docker = FakeClient()
    monkeypatch.setattr(L.backend, "_docker", docker)

    await L.backend.ensure_up()

    assert docker.posts == []           # already up -> no start issued
    assert L.backend.idle_since is None  # idle timer pushed out


async def test_ensure_up_slow_path_starts_and_waits(monkeypatch):
    L.backend._up = False
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/health": FakeResp(200)}))
    docker = FakeClient({"/start": FakeResp(204)})
    monkeypatch.setattr(L.backend, "_docker", docker)

    await L.backend.ensure_up()

    assert any("/start" in u for u, _ in docker.posts)
    assert L.backend._up is True


async def test_ensure_up_coalesces_concurrent_callers(monkeypatch):
    """Two concurrent wake requests must issue exactly one container start."""
    L.backend._up = False
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/health": FakeResp(200)}))
    docker = FakeClient({"/start": FakeResp(204)})
    monkeypatch.setattr(L.backend, "_docker", docker)

    await asyncio.gather(L.backend.ensure_up(), L.backend.ensure_up())

    starts = [u for u, _ in docker.posts if "/start" in u]
    assert len(starts) == 1


# --------------------------------------------------------------------------- #
# _probe + stale-_up self-heal
# --------------------------------------------------------------------------- #
async def test_probe_true_on_200(monkeypatch):
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/health": FakeResp(200)}))
    assert await L.backend._probe() is True


@pytest.mark.parametrize("resp", [
    pytest.param(FakeResp(500), id="non-200"),
    pytest.param(httpx.ConnectError("vllm down"), id="connection-error"),
])
async def test_probe_false_when_unhealthy(monkeypatch, resp):
    # _up is a cache that goes stale on an external stop/crash; _probe must return
    # False on both a non-200 and a connection error so ensure_up self-heals.
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/health": resp}))
    assert await L.backend._probe() is False


async def test_ensure_up_self_heals_when_up_cache_is_stale(monkeypatch):
    """_up=True but vLLM actually down (_probe False): ensure_up must NOT short-circuit
    on the stale cache -- it issues a start (the stale-wake-cache regression)."""
    L.backend._up = True
    L.backend.idle_since = 1234.0
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/health": FakeResp(500)}))
    docker = FakeClient({"/start": FakeResp(204)})
    monkeypatch.setattr(L.backend, "_docker", docker)

    async def _no_wait(_):  # we're testing the self-heal DECISION, not the wait loop
        return None
    monkeypatch.setattr(L.backend, "_wait_health", _no_wait)

    await L.backend.ensure_up()

    assert any("/start" in u for u, _ in docker.posts)  # self-healed, not skipped
    assert L.backend._up is True


# --------------------------------------------------------------------------- #
# _start_if_needed / _wait_health
# --------------------------------------------------------------------------- #
async def test_start_if_needed_raises_on_unknown_container(monkeypatch):
    docker = FakeClient({"/start": FakeResp(404)})
    monkeypatch.setattr(L.backend, "_docker", docker)

    with pytest.raises(RuntimeError):
        await L.backend._start_if_needed()


async def test_wait_health_raises_timeout(monkeypatch):
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/health": FakeResp(500)}))
    calls = {"n": 0}

    def fake_mono():  # deadline from call 1 (0.0); 2nd check still < deadline; 3rd exits
        calls["n"] += 1
        return 0.0 if calls["n"] <= 2 else 100.0

    # Replace the `time` NAME in the module (not the global time module): asyncio's loop
    # keeps the real monotonic clock, so only litellm_callbacks sees the fake.
    monkeypatch.setattr(L, "time", types.SimpleNamespace(monotonic=fake_mono))

    async def nosleep(_):
        return None

    monkeypatch.setattr(L.asyncio, "sleep", nosleep)

    with pytest.raises(TimeoutError):
        await L.backend._wait_health(1.0)


# --------------------------------------------------------------------------- #
# _backend_idle
# --------------------------------------------------------------------------- #
_METRICS_IDLE = (
    'vllm:num_requests_running{x="y"} 0\n'
    'vllm:num_requests_waiting{x="y"} 0\n'
    'vllm:num_requests_swapped{x="y"} 0\n'
)


async def test_backend_idle_true_when_all_zero(monkeypatch):
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/metrics": FakeResp(200, _METRICS_IDLE)}))
    assert await L.backend._backend_idle() is True


async def test_backend_idle_false_when_running(monkeypatch):
    text = _METRICS_IDLE.replace("num_requests_running{x=\"y\"} 0", "num_requests_running{x=\"y\"} 1")
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/metrics": FakeResp(200, text)}))
    assert await L.backend._backend_idle() is False


async def test_backend_idle_none_on_connection_error(monkeypatch):
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/metrics": httpx.ConnectError("down")}))
    assert await L.backend._backend_idle() is None


async def test_backend_idle_none_when_waiting_metric_missing(monkeypatch):
    text = 'vllm:num_requests_running{x="y"} 0\n'  # no waiting line
    monkeypatch.setattr(L.backend, "_client", FakeClient({"/metrics": FakeResp(200, text)}))
    assert await L.backend._backend_idle() is None


# --------------------------------------------------------------------------- #
# _stop
# --------------------------------------------------------------------------- #
async def test_stop_posts_stop_and_marks_down(monkeypatch):
    L.backend._up = True
    docker = FakeClient({"/stop": FakeResp(204)})
    monkeypatch.setattr(L.backend, "_docker", docker)

    await L.backend._stop()

    assert L.backend._up is False
    assert any("/stop" in u and p == {"t": 10} for u, p in docker.posts)


# --------------------------------------------------------------------------- #
# idle_watch (infinite loop — driven with a fake clock + sleep, exited via CancelledError)
# --------------------------------------------------------------------------- #
async def test_idle_watch_stops_after_sustained_idle(monkeypatch):
    L.backend._up = True
    monkeypatch.setattr(L, "IDLE_SECONDS", 10.0)
    monkeypatch.setattr(L, "POLL_SECONDS", 10.0)
    clock = {"t": 0.0}
    stopped = {"n": 0}

    async def fake_stop():
        stopped["n"] += 1
        L.backend._up = False
        L.backend.idle_since = None

    async def fake_sleep(s):
        clock["t"] += s
        if stopped["n"] >= 1:
            raise asyncio.CancelledError  # exits the infinite loop (BaseException, not caught)

    monkeypatch.setattr(L.backend, "_backend_idle", _atrue)
    monkeypatch.setattr(L.backend, "_stop", fake_stop)
    monkeypatch.setattr(L.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(L, "time", types.SimpleNamespace(monotonic=lambda: clock["t"]))

    with pytest.raises(asyncio.CancelledError):
        await L.backend.idle_watch()
    assert stopped["n"] == 1


async def test_idle_watch_never_stops_when_busy(monkeypatch):
    L.backend._up = True
    monkeypatch.setattr(L, "IDLE_SECONDS", 10.0)
    monkeypatch.setattr(L, "POLL_SECONDS", 5.0)
    stopped = {"n": 0}
    n = {"i": 0}

    async def fake_stop():
        stopped["n"] += 1

    async def fake_sleep(s):
        n["i"] += 1
        if n["i"] > 5:
            raise asyncio.CancelledError

    monkeypatch.setattr(L.backend, "_backend_idle", _afalse)
    monkeypatch.setattr(L.backend, "_stop", fake_stop)
    monkeypatch.setattr(L.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(L, "time", types.SimpleNamespace(monotonic=lambda: float(n["i"])))

    with pytest.raises(asyncio.CancelledError):
        await L.backend.idle_watch()
    assert stopped["n"] == 0
    assert L.backend.idle_since is None


async def test_idle_watch_preserves_state_when_metrics_unreadable(monkeypatch):
    L.backend._up = True
    L.backend.idle_since = 42.0
    monkeypatch.setattr(L, "IDLE_SECONDS", 10.0)
    monkeypatch.setattr(L, "POLL_SECONDS", 5.0)
    n = {"i": 0}

    async def fake_sleep(s):
        n["i"] += 1
        if n["i"] > 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(L.backend, "_backend_idle", _anone)
    monkeypatch.setattr(L.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(L, "time", types.SimpleNamespace(monotonic=lambda: float(n["i"])))

    with pytest.raises(asyncio.CancelledError):
        await L.backend.idle_watch()
    assert L.backend.idle_since == 42.0  # untouched when metrics unreadable


# --------------------------------------------------------------------------- #
# start_background_tasks (startup hook: registers Handler + spawns lifecycle tasks)
# --------------------------------------------------------------------------- #
async def test_start_background_tasks_registers_handler_and_is_idempotent(monkeypatch):
    import litellm

    async def block_forever():
        await asyncio.Event().wait()  # keep the spawned tasks pending so _tasks stays populated

    monkeypatch.setattr(L.backend, "bootstrap", block_forever)
    monkeypatch.setattr(L.backend, "idle_watch", block_forever)

    before = list(litellm.callbacks)
    try:
        await L.start_background_tasks()
        assert sum(1 for c in litellm.callbacks if isinstance(c, L.Handler)) == 1
        assert len(L._tasks) == 2

        await L.start_background_tasks()  # idempotent: no second Handler
        assert sum(1 for c in litellm.callbacks if isinstance(c, L.Handler)) == 1
    finally:
        tasks = list(L._tasks)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        L._tasks.clear()
        litellm.callbacks[:] = [c for c in litellm.callbacks if not isinstance(c, L.Handler)]


# --------------------------------------------------------------------------- #
# Handler: BOTH wake hooks exist and call ensure_up
# (the router path /v1/chat/completions AND the pass-through path /v1/messages).
# Deleting either is a silent regression: one completion path stops waking vLLM.
# --------------------------------------------------------------------------- #
async def test_handler_wakes_on_both_completion_paths(monkeypatch):
    h = L.Handler()
    for name in ("async_pre_call_hook", "async_pre_request_hook"):
        assert hasattr(h, name), f"Handler missing {name}; that completion path would not wake"

    wakes = {"n": 0}

    async def _count():
        wakes["n"] += 1

    monkeypatch.setattr(L.backend, "ensure_up", _count)

    await h.async_pre_call_hook({}, {}, {}, "acompletion")   # router path
    await h.async_pre_request_hook("m", [], {})              # /v1/messages pass-through
    assert wakes["n"] == 2
