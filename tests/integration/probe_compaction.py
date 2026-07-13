#!/usr/bin/env python3
"""Empirical compaction-trigger probe (integration harness, not a unit test).

Grows a Claude Code conversation (via `claude -p --resume`) with big user turns and
watches, per turn:
  * real per-turn input tokens  = delta of vLLM `vllm:prompt_tokens_total` (the truth)
  * claude's own reported usage  = stream-json assistant/result usage (what claude tracks)
  * compaction events           = any stream-json system event mentioning "compact"

Two layers can compact: claude's own proactive auto-compact (T4, gated by a per-session
breaker -- fires once) and the opt-in proxy-side polyfill (compact_20260112, fires on
every threshold crossing -- plan B). To ISOLATE the polyfill, run with a huge WINDOW so
claude's T4 threshold is unreachable (only the polyfill fires); the success signal is
REPEATED input-drops at ~PROXY_COMPACT_THRESHOLD with `compact=False` (the polyfill is
transparent -- no claude compact_boundary) and no 400.

Prerequisites to run live:
  1. `make start35` (stack up, vLLM warm).
  2. CLAUDE_QWEN_PROXY_COMPACT=1 on the litellm service (flip in docker-compose.moe.yaml
     + `make start35` to recreate), so the async_pre_request_hook injects context_management.
  3. `claude` on PATH (drives the proxy via ANTHROPIC_BASE_URL).
  4. vLLM :8000/metrics reachable from the host (the probe reads cumulative prompt_tokens).

Run directly (prints the human-readable table):
  python3 tests/integration/probe_compaction.py
  PROBE_WINDOW=1000000 PROBE_BLOB_TOKENS=18000 PROBE_TURNS=12 python3 tests/integration/probe_compaction.py

Or as a pytest integration test (skipped by default; env-gated):
  RUN_LIVE_COMPACTION_PROBE=1 make test-integration
  # or target it:
  RUN_LIVE_COMPACTION_PROBE=1 python3 -m pytest tests/integration/test_compaction_probe.py -m integration -o addopts="" -s
"""
import json
import os
import subprocess
import sys
import urllib.request

BASE_URL = os.environ.get("PROBE_BASE_URL", "http://localhost:4000")
VLLM_METRICS = os.environ.get("PROBE_VLLM_METRICS", "http://localhost:8000/metrics")


def _env_for_claude(window: int, pct: int) -> dict:
    env = dict(os.environ)
    env.update({
        "ANTHROPIC_BASE_URL": BASE_URL,
        "ANTHROPIC_AUTH_TOKEN": "sk-claude-qwen-local",
        "ANTHROPIC_MODEL": "qwen3.6-35b-a3b",
        "ANTHROPIC_SMALL_FAST_MODEL": "qwen3.6-35b-a3b-nothink",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(window),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "256",
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(pct),
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    })
    # The anti-tool-use prompt ("reply with exactly: ok") needs no tools; in -p mode
    # without --dangerously-skip-permissions tool use is auto-denied (no TTY), so the
    # auto-mode classifier's "Create Unsafe Agents" hard block is avoided.
    for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"):
        env.pop(k, None)
    return env


def metric() -> float | None:
    """Cumulative vllm:prompt_tokens_total, or None if /metrics is unreadable (tolerant:
    vLLM may be idle-stopped at the baseline read)."""
    try:
        for line in urllib.request.urlopen(VLLM_METRICS, timeout=5).read().decode().splitlines():
            if line.startswith("vllm:prompt_tokens_total{"):
                return float(line.split()[-1])
    except Exception:
        return None
    return 0.0


def blob(n_tokens: int) -> str:
    # ~4 chars/token; unique numbered lines so it tokenizes to ~n_tokens
    chars = n_tokens * 4
    out, i = [], 0
    while sum(len(l) for l in out) < chars:
        i += 1
        out.append(f"line {i:05d}: the quick brown fox jumps over the lazy dog near the riverbank token probe {i}")
    return "\n".join(out)


def run_turn(args, stdin_prompt, env, timeout=180):
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose"] + args + [stdin_prompt]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    events, compact_seen, usage = [], False, None
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        events.append(o)
        sub = o.get("subtype") or ""
        if o.get("type") == "system" and "compact" in str(sub).lower():
            compact_seen = True
        if o.get("type") == "assistant" and isinstance(o.get("message"), dict):
            u = o["message"].get("usage") or {}
            usage = u
    session_id = next((o.get("session_id") for o in events if o.get("session_id")), None)
    return {"compact": compact_seen, "usage": usage, "session_id": session_id,
            "events": events, "stderr": p.stderr[-300:], "ok": p.returncode == 0}


def run_probe(window: int, pct: int, turns: int, blob_tokens: int,
              turn_timeout: int = 180, verbose: bool = True) -> dict:
    """Grow a claude session and return structured per-turn results.

    Returns {"turns": [ {turn, delta, in, out, cache_r, cache_c, compact, ok, session_id,
    stderr} ... ], "baseline": float|None, "final": float|None}.
    """
    env = _env_for_claude(window, pct)
    baseline = metric()
    if verbose:
        print(f"WINDOW={window} PCT={pct} BLOB={blob_tokens}tok/turn TURNS={turns}")
        print(f"baseline cumulative prompt_tokens = {baseline}\n")
    prev = metric() or 0.0
    session_id = None
    big = blob(blob_tokens)
    records = []
    for t in range(1, turns + 1):
        args = [] if t == 1 else ["--resume", session_id]
        prompt = "reply with exactly: ok" if t == 1 else (
            f"Here is a document chunk to remember verbatim, then reply with exactly: ok\n\n{big}")
        try:
            r = run_turn(args, prompt, env, timeout=turn_timeout)
        except subprocess.TimeoutExpired as exc:
            records.append({"turn": t, "delta": None, "in": None, "out": None,
                            "cache_r": None, "cache_c": None, "compact": False,
                            "ok": False, "session_id": session_id,
                            "stderr": f"TIMEOUT after {turn_timeout}s: {exc}"})
            if verbose:
                print(f"turn {t:2d}: TIMEOUT after {turn_timeout}s; stopping")
            break
        if not session_id and r["session_id"]:
            session_id = r["session_id"]
        after = metric() or prev
        delta = after - prev
        prev = after
        u = r["usage"] or {}
        inp = u.get("input_tokens"); outp = u.get("output_tokens")
        cr = u.get("cache_read_input_tokens"); cc = u.get("cache_creation_input_tokens")
        rec = {"turn": t, "delta": delta, "in": inp, "out": outp,
               "cache_r": cr, "cache_c": cc, "compact": r["compact"],
               "ok": r["ok"], "session_id": session_id, "stderr": r["stderr"] if not r["ok"] else ""}
        records.append(rec)
        if verbose:
            print(f"turn {t:2d}: vLLM_prompt_delta={delta:7.0f}  "
                  f"claude_usage(in={inp} out={outp} cache_r={cr} cache_c={cc})  "
                  f"compact={r['compact']}  ok={r['ok']}  session={session_id}")
            if r["compact"]:
                print(f"  >>> CLAUDE COMPACTION EVENT at turn {t} (vLLM saw {delta:.0f} input tokens)")
        if not r["ok"]:
            if verbose:
                print(f"  >>> NON-ZERO EXIT: {r['stderr']}")
            break
    final = metric()
    if verbose:
        print(f"\nfinal cumulative prompt_tokens = {final}")
    return {"turns": records, "baseline": baseline, "final": final}


if __name__ == "__main__":
    WINDOW = int(os.environ.get("PROBE_WINDOW", "24000"))
    PCT = int(os.environ.get("PROBE_PCT", "80"))
    TURNS = int(os.environ.get("PROBE_TURNS", "12"))
    BLOB_TOKENS = int(os.environ.get("PROBE_BLOB_TOKENS", "6000"))
    run_probe(WINDOW, PCT, TURNS, BLOB_TOKENS)