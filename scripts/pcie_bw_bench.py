#!/usr/bin/env python3
"""Measure GPU<->host PCIe bandwidth on this box to bound the TP=2 all-reduce
cost on an x4 link, and estimate the SHM-fallback (worst-case) all-reduce
overhead for the Qwen3.6-27B hybrid model at single-stream (batch=1).

Why this grounds the TP=2 conclusion on a 1-GPU box:
  TP=2's all-reduce uses NCCL. Best case = P2P over PCIe; worst case (if P2P is
  blocked by ACS / topology / different root complexes) = SHM fallback through
  host memory, whose per-hop cost is exactly a cudaMemcpy D2H/H2D. If the worst
  case is negligible vs the decode step, the conclusion holds for BOTH
  transports, since P2P >= SHM throughput. The SHM number is a conservative
  upper bound on TP=2 comm overhead.

Model geometry (from config.json text_config): hidden_size=5120, num_hidden_layers=64.
  Per-layer all-reduce tensor (decode, batch=1, fp16) = hidden*2 = 10 KiB.
  Per-token one-way volume = 64 * 10 KiB = 640 KiB.
  SHM-fallback all-reduce moves ~2x the one-way volume through host memory
  (one D2H + one H2D of the one-way volume), i.e. ~1.28 MiB/token.

Reference step times come from scripts/coding_session_bench.py (~21 tps decode
-> ~47 ms/step; ~1026 prefill tps -> ~2 s for 2k tokens). Override via flags.
"""
import argparse
import math
import subprocess

import torch


def link_info():
    """Return (gen_cur, width_cur, gen_max, width_max) from nvidia-smi, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current,"
                          "pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"],
            text=True).strip()
        g, w, mg, mw = out.split(",")
        return g.strip(), w.strip(), mg.strip(), mw.strip()
    except Exception:
        return None


def ramp_link(payload_bytes=256 * 1024 * 1024, iters=3):
    """Do a few large transfers so PCIe ASPM ramps the link from idle Gen1 back
    to its operating speed before we query nvidia-smi for the link gen."""
    n = payload_bytes // 2
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dev = torch.empty(n, dtype=torch.float16, device="cuda")
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
        host.copy_(dev, non_blocking=True)
    torch.cuda.synchronize()
    del host, dev
    torch.cuda.empty_cache()


def bw(payload_bytes, direction, iters, warmup=3):
    """Peak D2H or H2D bandwidth for a given payload, using pinned + non_blocking copy."""
    n = payload_bytes // 2  # fp16 = 2 bytes/element
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dev = torch.empty(n, dtype=torch.float16, device="cuda")
    for _ in range(warmup):
        if direction == "H2D":
            dev.copy_(host, non_blocking=True)
        else:
            host.copy_(dev, non_blocking=True)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        if direction == "H2D":
            dev.copy_(host, non_blocking=True)
        else:
            host.copy_(dev, non_blocking=True)
    e.record()
    torch.cuda.synchronize()
    secs = s.elapsed_time(e) / 1000.0
    return payload_bytes * iters / secs  # bytes/sec


def fmt_gbs(bps):
    return f"{bps / 1e9:.2f} GB/s"


def bw_at(target, table):
    """Pick the measured payload whose log-size is closest to target."""
    best = min(table.keys(), key=lambda s: abs(math.log(s) - math.log(target)))
    return table[best], best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--prefill-tokens", type=int, default=2000)
    ap.add_argument("--decode-step-ms", type=float, default=47.0,
                    help="measured decode step time = 1000/out_tps (default ~21 tps)")
    ap.add_argument("--prefill-s", type=float, default=2.0,
                    help="measured prefill time for --prefill-tokens (default ~1026 tps)")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    ramp_link()  # wake PCIe ASPM so nvidia-smi reports the operating gen, not idle Gen1
    li = link_info()
    if li:
        print(f"PCIe link: gen {li[0]} (max {li[2]}), width x{li[1]} (max x{li[3]})")
    print(f"Model geometry: hidden={args.hidden}, layers={args.layers}, fp16 (2 bytes)")
    print(f"Reference: decode step {args.decode_step_ms:.0f} ms (~{1000/args.decode_step_ms:.1f} tps), "
          f"prefill {args.prefill_s:.1f}s for {args.prefill_tokens} tok")

    sizes = [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024, 640 * 1024, 1024 * 1024,
             4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024,
             256 * 1024 * 1024, 512 * 1024 * 1024, 1024 * 1024 * 1024]

    def n_iters(sz):
        if sz >= 256 * 1024 * 1024:
            return 5
        if sz >= 16 * 1024 * 1024:
            return 10
        return args.iters

    print(f"\n{'payload':>9} {'iters':>5} {'H2D':>10} {'D2H':>10}")
    print("-" * 38)
    h2d, d2h = {}, {}
    for sz in sizes:
        ni = n_iters(sz)
        b_h = bw(sz, "H2D", ni)
        b_d = bw(sz, "D2H", ni)
        h2d[sz], d2h[sz] = b_h, b_d
        print(f"{sz/1024:>8.0f}K {ni:>5} {fmt_gbs(b_h):>10} {fmt_gbs(b_d):>10}")

    # TP=2 all-reduce cost model (SHM fallback = worst case).
    per_layer_dec = args.hidden * 2
    per_tok_oneway = per_layer_dec * args.layers
    per_prefill_oneway = args.hidden * args.prefill_tokens * 2 * args.layers

    (bd, _) = bw_at(per_tok_oneway, d2h)
    (bh, _) = bw_at(per_tok_oneway, h2d)
    dec_ar_us = (per_tok_oneway / bd + per_tok_oneway / bh) * 1e6

    (pd, _) = bw_at(per_prefill_oneway, d2h)
    (ph, _) = bw_at(per_prefill_oneway, h2d)
    pre_ar_s = per_prefill_oneway / pd + per_prefill_oneway / ph

    dec_pct = 100 * dec_ar_us / (args.decode_step_ms * 1000)
    pre_pct = 100 * pre_ar_s / args.prefill_s

    print(f"\nTP=2 all-reduce, SHM-fallback (worst case), x{li[1] if li else '?'} gen{li[0] if li else '?'}:")
    print(f"  decode  : {per_tok_oneway/1024:.0f} KiB one-way/token -> "
          f"{dec_ar_us:7.1f} us/token = {dec_pct:5.2f}% of {args.decode_step_ms:.0f} ms step")
    print(f"  prefill : {per_prefill_oneway/1e9:.2f} GB one-way -> "
          f"{pre_ar_s*1000:7.1f} ms = {pre_pct:5.1f}% of {args.prefill_s:.1f} s prefill")
    print(f"\nWorst-case (SHM fallback) TP=2 comm overhead: {dec_pct:.2f}% decode, {pre_pct:.1f}% prefill.")
    print("P2P does one link hop (GPU0->GPU1) vs SHM's two (D2H+H2D), so if NCCL selects")
    print(f"P2P the overhead is roughly half: ~{dec_pct/2:.2f}% decode, ~{pre_pct/2:.1f}% prefill.")


if __name__ == "__main__":
    main()