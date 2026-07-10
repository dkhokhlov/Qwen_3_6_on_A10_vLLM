"""Integration: vLLM lifecycle under the live stack (self-heal after an outage).

DESTRUCTIVE: stops / SIGKILLs the vLLM container. Each test leaves the backend
running again at the end. Slow -- a cold wake reloads a multi-GB model (>= WAKE_TIMEOUT,
300s here). Skipped unless :4000 answers.

Run just this file:  make test-integration  (then)  or
  python3 -m pytest tests/integration/test_lifecycle.py -m integration -o addopts="" -s

Recovery model (verified on Docker 29.6.1):
  The system's REAL recovery path for a dead vLLM is LiteLLM wake-on-request: `ensure_up`
  resets its cached `_up` flag when `_probe` sees the backend down, then `docker start`s it.
  That works from BOTH a clean `docker stop` (SIGTERM) and a hard `docker kill` (SIGKILL,
  ExitCode 137 -- the OOM/crash case). We exercise both entry states below.

  Note on the restart policy: `vllm` ships `restart: unless-stopped` (its value is pinned by
  the unit test test_vllm_restart_is_unless_stopped), but empirically a `docker kill` here
  did NOT trigger a daemon restart -- RestartCount stayed 0 and the container sat exited
  until a request woke it. So wake-on-request is the mechanism we depend on; the restart
  policy is a backstop, not the tested mechanism.
"""
import os
import subprocess
import time

import pytest
import requests

pytestmark = pytest.mark.integration

WAKE_TIMEOUT = float(os.environ.get("VLLM_WAKE_TIMEOUT", "360"))


def _litellm_container() -> str:
    return subprocess.check_output(
        ["docker", "ps", "--filter", "name=litellm", "--format", "{{.Names}}"],
        text=True).strip().splitlines()[0]


def _vllm_container() -> str:
    out = subprocess.check_output(
        ["docker", "exec", _litellm_container(), "sh", "-c", "echo $BACKEND_CONTAINER"],
        text=True).strip()
    return out or "vllm-qwen35b"


def _is_running(name: str) -> bool:
    return subprocess.check_output(
        ["docker", "inspect", "-f", "{{.State.Running}}", name], text=True).strip() == "true"


def _wait_running(name: str, timeout: float = WAKE_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_running(name):
            return True
        time.sleep(5)
    return False


def _chat(litellm_base: str) -> int:
    names = {m["id"] for m in requests.get(f"{litellm_base}/models", timeout=10).json()["data"]}
    model = next(n for n in names if "-nothink" in n)
    r = requests.post(
        f"{litellm_base}/chat/completions",
        json={"model": model,
              "messages": [{"role": "user", "content": "Reply with the single word OK."}],
              "max_tokens": 8},
        timeout=WAKE_TIMEOUT)
    return r.status_code


@pytest.mark.parametrize("how", ["stop", "kill"], ids=["sigterm", "sigkill"])
def test_self_heal_after_external_outage(litellm_base, how):
    """Take vLLM down out-of-band, then one request must self-heal it: ensure_up sees the
    stale `_up` + a failed probe and wakes the backend. Covers both a clean stop (`docker
    stop` = SIGTERM) and a hard crash (`docker kill` = SIGKILL, ExitCode 137)."""
    vllm = _vllm_container()
    subprocess.check_call(["docker", how, vllm])
    assert not _is_running(vllm)
    assert _chat(litellm_base) == 200          # litellm ensure_up wakes the dead backend
    assert _wait_running(vllm)
