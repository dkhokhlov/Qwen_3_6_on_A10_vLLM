#!/usr/bin/env python3
"""Simulate a growing coding chat session against the LiteLLM proxy endpoint and
measure per-turn prefill TPS and output TPS.

Each turn appends a ~2k-token user message and requests ~500 completion tokens.
Prior turns become a shared prefix, so if prefix caching is effective the
*uncached* prefill cost should stay roughly flat even as the total prompt grows.

Cache hit is read from vLLM /metrics (vllm:prefix_cache_{queries,hits}_total
deltas), which is authoritative and version-stable. Token counts come from the
streaming `usage` chunk.

Usage:
    python3 scripts/coding_session_bench.py --turns 8
"""
import argparse
import json
import time
import sys
from urllib.parse import urlparse, urlunparse

import requests


def metrics_url_for(base_url):
    """vLLM serves /metrics at the root, not under /v1."""
    p = urlparse(base_url)
    return urlunparse((p.scheme, p.netloc, "/metrics", "", "", ""))


def build_user_message(turn, target_chars):
    """~2k tokens of coding-task text, unique per turn so the new portion is
    real prefill (not a verbatim repeat of a prior turn)."""
    spec = (
        f"Turn {turn} feature request: refactor the order-matching engine. "
        "Add a rate limiter around the matching loop, guard against self-trades, "
        "and emit a fill event carrying price, size, and taker/maker flags. "
        "Existing code:\n"
        "    def match(book):\n"
        "        for bid, ask in book.pairs():\n"
        "            if bid.price >= ask.price:\n"
        "                execute(bid, ask)\n"
        "Modify it to add the limiter, the self-trade guard, and the fill event. "
        "Keep it deterministic and side-effect free where possible. "
    )
    reps = max(1, target_chars // len(spec) + 1)
    return (spec * reps)[:target_chars]


def get_cache_counters(base_url, timeout):
    """Return (queries_total, hits_total) from /metrics, or (None, None) on failure."""
    try:
        r = requests.get(metrics_url_for(base_url), timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return None, None
    q = h = None
    for line in r.text.splitlines():
        if line.startswith("vllm:prefix_cache_queries_total{"):
            q = float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_hits_total{"):
            h = float(line.split()[-1])
    return q, h


def stream_turn(base_url, model, messages, max_tokens, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    t_first = None
    usage = None
    pieces = []
    with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    pieces.append(piece)
    t_end = time.perf_counter()
    return {
        "t_first": (t_first - t0) if t_first else None,
        "t_total": t_end - t0,
        "usage": usage,
        "content": "".join(pieces),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Through the LiteLLM sole proxy (vLLM is internal-only); the -nothink model
    # variant skips reasoning, so decode/prefill TPS are measured without thinking tokens.
    # Cache counters are read from /metrics; vLLM's /metrics is not proxied by LiteLLM,
    # so when base-url is the proxy they come back absent -> hit% reads 0 (TPS still valid).
    ap.add_argument("--base-url", default="http://localhost:4000/v1")
    ap.add_argument("--model", default="qwen3.6-27b-nothink")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--input-tokens", type=int, default=2000,
                    help="target size of each new user message (approx)")
    ap.add_argument("--max-tokens", type=int, default=500,
                    help="completion cap per turn (thinking disabled)")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    target_chars = args.input_tokens * 4  # ~4 chars/token for English/code
    system = ("You are a senior production engineer. Reply with concise, "
              "compilable code only, no prose.")
    messages = [{"role": "system", "content": system}]

    print(f"# growing coding session: {args.turns} turns, "
          f"~{args.input_tokens} tok input/turn, {args.max_tokens} tok output/turn, "
          f"model={args.model}")
    header = (f"{'turn':>4} {'prompt':>7} {'out':>5} {'seq':>7} {'cached':>7} {'uncached':>8} "
              f"{'ttft_s':>7} {'prefill_tps':>11} {'out_tps':>8} {'lat_s':>6} {'hit%':>5}")
    print(header)
    print("-" * len(header))

    for turn in range(1, args.turns + 1):
        user = build_user_message(turn, target_chars)
        messages.append({"role": "user", "content": user})

        q0, h0 = get_cache_counters(args.base_url, args.timeout)
        res = stream_turn(args.base_url, args.model, messages,
                          args.max_tokens, args.timeout)
        q1, h1 = get_cache_counters(args.base_url, args.timeout)

        u = res["usage"] or {}
        prompt_tok = u.get("prompt_tokens", 0)
        comp_tok = u.get("completion_tokens", 0)

        cached = int(h1 - h0) if (h0 is not None and h1 is not None) else 0
        # metrics query delta should ~= prompt_tok; fall back to that if absent
        if q0 is None or q1 is None:
            cached = 0
        uncached = max(prompt_tok - cached, 0)

        ttft = res["t_first"] or 0.0
        gen_dur = max(res["t_total"] - ttft, 1e-6)
        prefill_tps = (uncached / ttft) if ttft > 1e-6 else 0.0
        out_tps = comp_tok / gen_dur
        hit_pct = (100.0 * cached / prompt_tok) if prompt_tok else 0.0
        seq_tok = prompt_tok + comp_tok

        print(f"{turn:>4} {prompt_tok:>7} {comp_tok:>5} {seq_tok:>7} {cached:>7} {uncached:>8} "
              f"{ttft:>7.2f} {prefill_tps:>11.1f} {out_tps:>8.1f} "
              f"{res['t_total']:>6.1f} {hit_pct:>4.0f}")
        sys.stdout.flush()

        # Append the REAL assistant reply so the shared prefix matches the
        # actual token sequence (a placeholder would diverge and miss cache).
        messages.append({"role": "assistant", "content": res["content"]})


if __name__ == "__main__":
    main()