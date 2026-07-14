"""Unit tests for the pure helpers in tests/integration/probe_compaction.py (the live
probe itself is integration-only / env-gated / CI-excluded). metric() reads vLLM
/metrics; it MUST return None -- not 0.0 -- when the prompt_tokens line is absent, so
callers' `metric() or prev` carry-forward distinguishes "couldn't read" from a real
cumulative count (a cumulative counter is never genuinely 0.0 after the first request;
conflating absent-with-0.0 masked transient empty reads as delta=0)."""
import pathlib
import sys

# tests/integration has no __init__.py (pytest prepend import mode); add it to sys.path
# so this unit test can import the probe helper module directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "integration"))
import probe_compaction as P  # noqa: E402


class _FakeResp:
    def __init__(self, body): self._body = body
    def read(self): return self._body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_metric_returns_none_when_prompt_tokens_line_absent(monkeypatch):
    # /metrics readable but no vllm:prompt_tokens_total line (vLLM just woke, served
    # nothing) -> None, NOT 0.0.
    body = b"# HELP vllm:foo\nvllm:foo 5\n"  # no prompt_tokens_total line
    monkeypatch.setattr(P.urllib.request, "urlopen", lambda url, timeout=5: _FakeResp(body))
    assert P.metric() is None


def test_metric_returns_float_when_line_present(monkeypatch):
    body = b'vllm:prompt_tokens_total{model="qwen"} 538\n'
    monkeypatch.setattr(P.urllib.request, "urlopen", lambda url, timeout=5: _FakeResp(body))
    assert P.metric() == 538.0


def test_metric_returns_none_on_exception(monkeypatch):
    monkeypatch.setattr(
        P.urllib.request, "urlopen",
        lambda url, timeout=5: (_ for _ in ()).throw(OSError("down")))
    assert P.metric() is None