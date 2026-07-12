"""Unit tests for the pure helpers in coding_session_bench. Loads the script by file
path (it isn't a package) and exercises only the no-network / no-GPU functions.

The pcie_bw_bench helpers live in tests/gpu/ (run inside the vLLM image via
`make test-pcie`): that script imports torch, which has no place in this hermetic suite."""
import importlib.util
from pathlib import Path

import pytest
import requests

REPO = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


csb = _load_module("scripts/coding_session_bench.py", "coding_session_bench")


# --------------------------------------------------------------------------- #
# coding_session_bench.metrics_url_for: rewrite /v1 base -> root /metrics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base, expected", [
    ("http://localhost:8000/v1", "http://localhost:8000/metrics"),
    ("http://localhost:4000/v1", "http://localhost:4000/metrics"),
    ("http://h:8000/v1/chat/completions", "http://h:8000/metrics"),  # path+query stripped
])
def test_metrics_url_for(base, expected):
    assert csb.metrics_url_for(base) == expected


# --------------------------------------------------------------------------- #
# coding_session_bench.build_user_message: deterministic, target-length, per-turn unique
# --------------------------------------------------------------------------- #
def test_build_user_message_length_and_turn_token():
    msg = csb.build_user_message(1, 2000)
    assert len(msg) == 2000
    assert "Turn 1" in msg


def test_build_user_message_deterministic():
    assert csb.build_user_message(3, 1500) == csb.build_user_message(3, 1500)


def test_build_user_message_differs_per_turn():
    assert "Turn 2" in csb.build_user_message(2, 2000)
    assert csb.build_user_message(1, 2000) != csb.build_user_message(2, 2000)


def test_build_user_message_small_target():
    msg = csb.build_user_message(1, 50)
    assert len(msg) == 50
    assert "Turn 1" in msg


# --------------------------------------------------------------------------- #
# coding_session_bench.get_cache_counters: parse prefix-cache metrics (network mocked)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.RequestException(f"status {self.status_code}")


_METRICS_TEXT = (
    'vllm:prefix_cache_queries_total{x="y"} 100\n'
    'vllm:prefix_cache_hits_total{x="y"} 42\n'
)


def test_get_cache_counters_parses(monkeypatch):
    monkeypatch.setattr(csb.requests, "get", lambda *a, **k: _FakeResp(_METRICS_TEXT))
    assert csb.get_cache_counters("http://h:8000/v1", 5) == (100.0, 42.0)


def test_get_cache_counters_missing_metric_is_none(monkeypatch):
    monkeypatch.setattr(csb.requests, "get", lambda *a, **k: _FakeResp("unrelated\n"))
    q, h = csb.get_cache_counters("http://h:8000/v1", 5)
    assert (q, h) == (None, None)


def test_get_cache_counters_failure_returns_none_pair(monkeypatch):
    def boom(*a, **k):
        raise requests.RequestException("nope")

    monkeypatch.setattr(csb.requests, "get", boom)
    assert csb.get_cache_counters("http://h:8000/v1", 5) == (None, None)
