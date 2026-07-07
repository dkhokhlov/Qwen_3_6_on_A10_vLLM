# Running Qwen3.6 on NVIDIA A10 24GB with vLLM

Configs and measurements for both Qwen3.6 checkpoints on a single
[NVIDIA A10](https://www.techpowerup.com/gpu-specs/a10-pcie.c3793) (24GB) with vLLM:

- **[Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ) — a
  Dense model** (27B active params, no experts) at full **64k context, no CPU
  offload**, output `21tps` flat to max context.
- **[Qwen3.6-35B-A3B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)
  — a MoE model** (35B total / 3B active, 256 routed experts) at full **128k
  context with 2.2 GiB UVA CPU offload**, output `15.4tps` flat to max context.

**The bigger MoE serves *more* context than the smaller dense model** — 128k vs
64k — because MoE has a smaller KV per token (fewer full-attn layers, smaller
hidden dim). The MoE's 35B weights don't fit without offload; the dense 27B fits
cleanly. See [Dense vs MoE — why the bigger model serves more context](#dense-vs-moe--why-the-bigger-model-serves-more-context).

- vLLM OpenAI API: `http://localhost:8000/v1`
  - 27B served-model-name: `qwen3.6-27b`
  - 35B MoE served-model-name: `qwen3.6-35b-a3b`
- open-webui: `http://localhost:3000/`

## TOC

- [Quick Start](#quick-start)
- [Repo structure](#repo-structure)
- [TL;DR — one A10, both models](#tldr--one-a10-both-models)
- [Dense vs MoE — why the bigger model serves more context](#dense-vs-moe--why-the-bigger-model-serves-more-context)
- [Hardware](#hardware)
- [Deployment A — Qwen3.6-27B-AWQ (Dense)](#deployment-a--qwen36-27b-awq-dense)
- [Deployment B — Qwen3.6-35B-A3B-AWQ (MoE)](#deployment-b--qwen36-35b-a3b-awq-moe)
- [Prefix caching](#prefix-caching)
- [PCIe bandwidth — grounding the TP=2-on-x4 question](#pcie-bandwidth--grounding-the-tp2-on-x4-question)
- [Two A10 cards — TP=2 projections for Dense and MoE](#two-a10-cards--tp2-projections-for-dense-and-moe)
- [Experiments that did NOT work (do not retry without reason)](#experiments-that-did-not-work-do-not-retry-without-reason)
- [Hardware ceiling facts](#hardware-ceiling-facts)
- [Gotchas](#gotchas)

## Quick Start

| target | what |
|---|---|
| `make` | help (default) |
| `make run` | `docker compose up` — 27B Dense, foreground (Ctrl-C to stop) |
| `make start` | `docker compose up -d` — 27B Dense, detached |
| `make stop` | `docker compose stop` — 27B Dense |
| `make run35` | `docker compose -f docker-compose.moe.yaml up` — 35B MoE, foreground |
| `make start35` | `docker compose -f docker-compose.moe.yaml up -d` — 35B MoE, detached |
| `make stop35` | `docker compose -f docker-compose.moe.yaml stop` — 35B MoE |
| `make bench` | growing coding-session bench to ~64k context on the 27B (override `TURNS=N`) |
| `make bench35` | same bench on the 35B MoE (override `TURNS=N`; `TURNS=54` → ~128k) |
| `make bench_pcie` | GPU↔host PCIe bandwidth (needs a free GPU: `make stop` first) |

`run`/`start`/`stop` target the 27B Dense stack (`docker-compose.yaml`);
`run35`/`start35`/`stop35` target the 35B MoE stack (`docker-compose.moe.yaml`).
The two stacks use the same port (8000) and container name prefix, so stop one
before starting the other.

The coding-session bench (`scripts/coding_session_bench.py`) streams a chat that
grows one ~2k-token user message + ~500-token assistant reply per turn and
reports per-turn prompt/seq/cached/uncached tokens, TTFT, prefill TPS, output
TPS, and prefix-cache hit %. Cache hit is read from vLLM `/metrics` (at the
**root**, not `/v1`).

## Repo structure

| file | what |
|---|---|
| `docker-compose.yaml` | vLLM (27B Dense) + open-webui stack |
| `docker-compose.moe.yaml` | vLLM (35B MoE, 128k, 2.2 GiB offload) stack |
| `Makefile` | `run` / `start` / `stop` / `run35` / `start35` / `stop35` / `bench` / `bench35` / `bench_pcie` |
| `scripts/coding_session_bench.py` | growing coding-session bench (prefill/output tps, cache hit from `/metrics`) |
| `scripts/pcie_bw_bench.py` | GPU↔host PCIe D2H/H2D bandwidth + TP=2 all-reduce estimate |
| `README.md` | this doc |

## TL;DR — one A10, both models

| | Qwen3.6-27B-AWQ (**Dense**) | Qwen3.6-35B-A3B-AWQ (**MoE**) |
|---|---|---|
| architecture | hybrid: 48 GDN linear + 16 full-attn / 64 layers | hybrid: 30 GDN linear + 10 full-attn / 40 layers |
| total params | 27B | 35B |
| **active params / token** | **27B (all)** | **3B (1 shared + routed experts)** |
| hidden dim | 5120 | 2048 |
| full-attn layers | 16 | 10 (1-in-4) |
| attn / KV heads / head_dim | — | 16 / 2 / 256 |
| routed experts | none | 256 + 1 shared |
| weights on GPU (AWQ-Marlin) | 18.83 GiB | ~21.5 GiB (language-only) |
| KV+state per token | large (16 full-attn × 5120) | small (10 full-attn × 2048) |
| CPU offload | none | **2.2 GiB (UVA)** |
| `--max-model-len` (1× A10) | **64 000** | **128 000** |
| KV-cache tokens | 64 744 | 150 349 |
| decode tps (flat to max ctx) | ~21 → 19.6 (−10%) | 15.3 → 15.4 (flat) |
| prefill tps (short → long ctx) | 1026 → 526 (−49%) | 966 → 698 (−28%) |
| prefix-cache block (align) | 800 tokens | 1072 tokens |
| `--gpu-memory-utilization` | 0.97 | 0.95 |
| `--max-num-batched-tokens` | 1024 | 1280 |
| compose file | `docker-compose.yaml` | `docker-compose.moe.yaml` |
| bench target | `make bench` | `make bench35` |

## Dense vs MoE — why the bigger model serves more context

**The 35B model serves 128k context while the 27B only serves 64k — despite
the 35B having more weights.** Three facts invert the "bigger model = less
context" intuition:

| lever | Dense 27B | MoE 35B-A3B | effect on context ceiling |
|---|---|---|---|
| active params / token | 27B (every MLP runs) | 3B (only routed experts run) | MoE: lighter compute, but **irrelevant to context** (context is KV-bound, not compute-bound) |
| full-attn layers | 16 | 10 (1-in-4) | MoE has fewer layers attending over the prefix |
| hidden dim | 5120 | 2048 | MoE KV head is 4× smaller per layer |
| **KV + state per token** | **~35 KiB** (large; ~3× MoE) | **~11 KiB** (but far fewer full-attn tokens) | MoE pays less context cost per token |
| weights on GPU | 18.83 GiB (fits) | ~21.5 GiB (does **not** fit) | MoE must offload 2.2 GiB |

The dense 27B is **param-light enough to fit cleanly** (18.83 GiB leaves 3.23
GiB for KV+state → 64k) but its 16 full-attn layers × hidden 5120 make each
context token expensive, so 64k is the no-offload ceiling.

The MoE 35B is **param-heavy on weights** (21.5 GiB > free GPU) but its 10
full-attn layers × hidden 2048 make each context token cheap. Once 2.2 GiB of
the weights is offloaded to host RAM (UVA — zero-copy GPU-direct PCIe read),
the freed GPU holds ~150k tokens of KV+state → 128k context with margin.

**MoE has no free lunch:** the 2.2 GiB offload is a serving-time tax — a
100 %-CPU UVA offloader thread and the GPU PCIe-stalled to ~80 % power
(120/150 W). See [Offload efficiency cost](#offload-efficiency-cost). The dense
27B pays none of this.

### MoE no-offload math (why offload is required, not optional)

Budget at util 0.97 = 0.97 × 22.06 = 21.40 GiB. Post-repack weights ≈ 21.38 GiB.
From the 1 GiB offload run: GPU weights 20.31, KV 0.65 → vLLM's non-KV reserve
= (21.40 − 20.31) − 0.65 ≈ 0.44 GiB (mamba state + activation scratch + safety).

| path | weights | reserve | total | vs 21.40 GiB budget | usable context |
|---|---|---|---|---|---|
| GPU-only @ 0.97 | 21.38 | 0.44 | 21.82 | over by 0.42 (Marlin repack OOM) | ~0 |
| GPU-only @ 0.99 | 21.38 | 0.44 | 21.82 | over by ~0.02 (but 0.99 fails init) | ~2k |
| **2.2 GiB offload @ 0.95** | 19.18 | 0.44 | 19.62 | 1.78 GiB for KV+state | **128k** |

So GPU-only is param+reserve-bound by ~0.2 GiB at the achievable util; offload
is required for any usable context. (The offload is a *serving-time* offload —
it persists into serving, not load-only — but the penalty is ~15 tps, not 2 tps.)

## Hardware

| | |
|---|---|
| GPU | NVIDIA A10, Ampere sm86, 24GB VRAM (~22.06 GiB usable by vLLM) |
| PCIe | Gen4 **x4** — eGPU rig: A10 in an external enclosure, host↔GPU routed via an M.2 NVMe → OCuLink (SFF-8612) adapter, which exposes only 4 lanes. Card is x16-capable; the adapter/wiring caps the link at x4. |
| TP | 1 (single GPU) |
| 27B weights on GPU | 18.83 GiB |
| 27B KV + state cache | ~2.15 GiB → 64 744 tokens |
| 35B MoE weights on GPU | ~19.2 GiB (after 2.2 GiB offload of 21.5 GiB) |
| 35B MoE KV + state cache | ~1.78 GiB → 150 349 tokens |

A second A10 on the same host would lift both ceilings and remove the MoE
offload tax — see [Two A10 cards — TP=2 projections](#two-a10-cards--tp2-projections-for-dense-and-moe).

## Deployment A — Qwen3.6-27B-AWQ (Dense)

### Why it fits on 24GB: AWQ + hybrid linear attention + tuned caches

Three things together make 64k context on a 24GB GPU possible — and keep decode
fast at long context.

#### 1. AWQ INT4 weights on the fast Marlin kernel

`QuantTrio/Qwen3.6-27B-AWQ` is genuinely packed INT4 (not just a name), and vLLM
runs it on **AWQ-Marlin** (fused dequant + INT4 GEMM) — *not* dequantized to FP16.

| | size | what |
|---|---|---|
| Pure INT4 (theoretical) | ~13.5 GiB | 27B × 0.5 bytes/param |
| On-disk checkpoint | 20.35 GiB | INT4 + FP16 leftovers + visual branch + AWQ scale/zp |
| **Loaded on GPU** | **18.83 GiB** | `--language-model-only` skips ~1.5 GiB visual branch |

The ~5 GiB gap between pure-INT4 and loaded = the **FP16 leftovers** kept
unquantized for accuracy (config `modules_to_not_convert`):

| kept FP16 | quantized INT4 (AWQ-Marlin) |
|---|---|
| `self_attn.{q,k,v}_proj`, `linear_attn.in_proj_{a,b}`, `model.layers.0`, `mtp` | MLP `gate` / `up` / `down` (the big tensors) |

If vLLM dequantized to FP16 the footprint would be ~54 GiB — wouldn't fit, and
decode would be ~3× slower. Marlin keeps INT4 packed in GDDR6 and dequants inside
the GEMM, so you get the full memory/bandwidth benefit.

#### 2. Hybrid architecture — 48 of 64 layers are linear (GatedDeltaNet)

This is **why output tps stays flat as context grows.** Only 16 layers are
full-attention (KV-cached, `O(context)` per token); 48 are linear-attention with
a fixed-size SSM state (`O(1)` per token). 75 % of layers don't pay the
growing-context cost.

```
Qwen3.6-27B-AWQ  —  64 transformer layers (hybrid)

  ┌──────────────────────────────────────────────┐
  │  16 × full-attention layers                  │  KV cache (FP8)
  │     self_attn q/k/v  — FP16                  │  O(context) / token
  │     MLP gate/up/down — INT4 AWQ-Marlin       │  ← grows with context
  └──────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────┐
  │  48 × GatedDeltaNet linear-attention layers  │  SSM state cache (FP16)
  │     linear_attn in_proj — FP16               │  O(1) / token
  │     MLP gate/up/down   — INT4 AWQ-Marlin     │  ← fixed cost at any ctx
  └──────────────────────────────────────────────┘

  + 1 MTP head (present, unused — see Experiments)
  + visual branch (skipped via --language-model-only)
```

Measured across a growing session (2k → 64k context), **decode barely moves**:

| context | out_tps (decode) | vs 2k | prefill_tps (uncached new turn) |
|---|---|---|---|
| 2k | 21.9 | — | 1026 |
| 18k | 20.9 | −5 % | 802 |
| 40k | 20.2 | −8 % | 646 |
| 64k | 19.6 | **−10 %** | 526 |

Decode slows only ~10 % over 62k tokens of context because the 48 linear layers
are `O(1)`. Prefill of the new turn slows ~49 % because the 16 full-attention
layers attend over the entire cached prefix per prefilled token. See
[Verified performance (27B)](#verified-performance-27b).

#### 3. KV + state caches tuned for the hybrid mix

| cache | dtype | covers | why |
|---|---|---|---|
| KV cache | **fp8** | 16 full-attn layers | FlashInfer on Ampere; halves KV in GDDR6 |
| mamba state cache | **float16** | 48 linear-attn layers | **key lever** — config defaults FP32; FP16 lifts the no-offload ceiling 54.9k → 72.5k tokens |
| prefix cache | on (align mode) | cross-request reuse | experimental for hybrid; 800-token block size |

Without the `--mamba-ssm-cache-dtype float16` lever, 64k context would not fit
on this GPU. See [Working config (27B)](#working-config-27b).

### Working config (27B)

`docker-compose.yaml` vLLM command — bulleted:

- `--model QuantTrio/Qwen3.6-27B-AWQ`
- `--served-model-name qwen3.6-27b`
- `--language-model-only`
- `--tensor-parallel-size 1`
- `--gpu-memory-utilization 0.97`
- `--max-model-len 64000`
- `--kv-cache-dtype fp8`
- `--mamba-cache-dtype float16`
- `--mamba-ssm-cache-dtype float16`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 1024`
- `--enforce-eager`
- `--enable-prefix-caching`
- `--enable-auto-tool-choice`
- `--tool-call-parser hermes`
- `--reasoning-parser qwen3`
- `--trust-remote-code`

Flag rationale (non-obvious ones):

| flag | value | why |
|---|---|---|
| `--language-model-only` | — | serve the text branch of the VLM checkpoint (skip visual) |
| `--gpu-memory-utilization` | 0.97 | 0.98 + 70k OOM'd on prefill (zero activation headroom); 0.97 leaves ~220 MiB |
| `--max-model-len` | 64000 | stable ceiling; align-mode (prefix caching) reduces KV so 65536 → 64000 |
| `--kv-cache-dtype` | fp8 | quantize the 16 full-attn layers' KV (FlashInfer on Ampere) |
| `--mamba-ssm-cache-dtype` | **float16** | **key lever** — FP32→FP16 on 48 linear layers; ceiling 54.9k → 72.5k tokens |
| `--mamba-cache-dtype` | float16 | FP16 mamba cache |
| `--max-num-seqs` | 1 | single-user |
| `--max-num-batched-tokens` | 1024 | halves prefill activation spike (2048 OOM'd) |
| `--enforce-eager` | — | no CUDA graph capture; saves memory (see Experiments) |
| `--enable-prefix-caching` | — | experimental hybrid support; align mode, 800-token block |
| `--reasoning-parser` | qwen3 | thinking → `message.reasoning`, not eating `max_tokens` |
| `--tool-call-parser` | hermes | tool-call parsing |
| `--trust-remote-code` | — | hybrid arch needs it |

Env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`. HF cache mounted read-only; `HF_TOKEN` resolves from
host env (not committed).

### Verified performance (27B)

`make bench` (default `TURNS=27`) grows the session to seq 63924 ≈ the 64k
ceiling. Condensed (every 8th turn + final):

```
turn  prompt   out     seq  cached uncached  ttft_s prefill_tps  out_tps  lat_s  hit%
   1    1885   474    2359       0     1885    1.84      1026.5     21.9   23.5    0
   8   18369   500   18869   16000     2369    2.96       801.7     20.9   26.8   87
  17   39704   500   40204   36800     2904    4.49       646.2     20.2   29.2   93
  27   63424   500   63924   60800     2624    4.99       525.8     19.6   30.5   96
```

- **Output TPS ~21 → 19.6** — GDDR6-bandwidth-bound (~600 GB/s, 18.83 GiB weights).
  Stays nearly flat to 64k because 48/64 layers are linear (`O(1)`/token).
- **Prefill TPS 1026 → 526** (uncached new turn) — the 16 full-attn layers
  attend over the growing prefix per prefilled token (`O(context × uncached)`).
- **Latency 23.5 → 30.5 s** (+30 % over 62k tokens); decode is ~84 % of latency
  at max context (`500/19.6 = 25.5 s` of the 30.5 s turn). Modest growth because
  the dominant decode term barely moves.
- **Prefix caching holds at long context** — hit 96 % at 64k (cached = 60 800 =
  76 × 800-token blocks); only the new turn is prefilled.

> Note on turn 1: a cold start gives 0 % hit (ttft ~1.8 s). Re-running with the
> same user text warms the prefix cache and turn 1 hits ~85 % (ttft ~0.3 s). The
> table above is a cold start.

---

## Deployment B — Qwen3.6-35B-A3B-AWQ (MoE)

### Architecture

```
Qwen3.6-35B-A3B-AWQ  —  40 transformer layers (hybrid)

  ┌──────────────────────────────────────────────┐
  │  10 × full-attention layers (1-in-4)         │  KV cache (FP8)
  │     self_attn q/k/v  — FP16                  │  O(context) / token
  │     MLP: 256 routed + 1 shared expert        │  ← grows with context
  │          only ~8 routed + shared active/token│
  └──────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────┐
  │  30 × GatedDeltaNet linear-attention layers  │  SSM state cache (FP16)
  │     linear_attn in_proj — FP16               │  O(1) / token
  │     MLP gate/up/down   — INT4 AWQ-Marlin     │  ← fixed cost at any ctx
  └──────────────────────────────────────────────┘

  arch: Qwen3_5MoeForConditionalGeneration
  attn heads 16 / KV heads 2 / head_dim 256
  active params/token ≈ 3B  (35B total weights)
  + visual branch (333 keys) + MTP (785 keys) — skipped via --language-model-only
```

The MoE keeps the same hybrid linear/full-attn split as the dense 27B but with
**fewer layers (40 vs 64), fewer full-attn layers (10 vs 16), and a 4× smaller
hidden (2048 vs 5120)** — that is the source of its larger context ceiling.

### Why offload is required (measured envelope)

QuantTrio's AWQ→Marlin MoE repack (`awq_marlin_moe_repack`) needs ~128 MiB
scratch with raw+Marlin coexisting; with 0 offload the 21.4 GiB weights leave no
room → OOM at 21.64 GiB. `--cpu-offload-gb N` moves N GiB of weights to CPU
*before* the repack (UVA offloader), so GPU weights = 21.38 − N, the repack
fits, and the model loads. The offload persists into serving — but the penalty
is small (see [Offload efficiency cost](#offload-efficiency-cost)).

Measured envelope (QuantTrio, vLLM nightly, `--language-model-only`,
`--enforce-eager`, fp8 KV, fp16 mamba state):

| cpu-offload-gb | util | max-model-len | prefix-cache | batched | GPU KV tokens | decode tps | status |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.97 | 60 000  | on | 2048 | 60 967  | ~24 | short-prompt only (NOT prefill-tested) |
| 2.0 | 0.98 | 160 000 | on | 2048 | 181 538 | —   | OOM first prefill (util 0.98 over-sizes KV) |
| 2.0 | 0.97 | 140 000 | on | 2048 | 158 394 | ~17 | short-prompt only — OOMs on real prefill |
| 2.2 | 0.97 | 160 000 | on | 2048 | 192 820 | ~14 | short-prompt only — OOMs on real prefill |
| **2.2** | **0.95** | **128 000** | **on** | **1280** | **150 349** | **~15.5** | **WORKS — real prefill to 128k, prefix-cache 98 % hit** |

> The top 4 rows only tested a 16-token prompt; they OOM on the first real
> (>1k-token) prefill. Root cause: vLLM under-reserves runtime overhead for this
> hybrid MoE — it estimates ~0.30 GiB reserve but the real overhead is ~0.93 GiB
> (mamba state + FlashInfer 394 MiB workspace + 1072-token mamba-page block
> padding + CUDA ctx). So vLLM over-sizes KV and leaves no room for the prefill
> activation spike (~0.4 GiB for a 2048-token chunk). OOM signature:
> `12.75 MiB free, tried to allocate 20 MiB` on the first forward.

Three levers, all required, make the 128k row work:

| lever | value | role |
|---|---|---|
| `--gpu-memory-utilization` | **0.95** | forces conservative KV sizing → leaves prefill-spike headroom (0.97 over-sizes KV → OOM even with prefix cache off) |
| `--max-num-batched-tokens` | **1280** | ≥ the mamba align-mode block size 1072 (assertion `block_size <= max_num_batched_tokens` fires if smaller) and small enough to bound the prefill spike |
| `--cpu-offload-gb` | **2.2** | frees the GPU headroom that lets Marlin repack + 128k KV both fit |

### Working config (35B MoE)

`docker-compose.moe.yaml` vLLM command — bulleted:

- `--model QuantTrio/Qwen3.6-35B-A3B-AWQ`
- `--served-model-name qwen3.6-35b-a3b`
- `--language-model-only`
- `--host 0.0.0.0`
- `--port 8000`
- `--tensor-parallel-size 1`
- `--gpu-memory-utilization 0.95`
- `--cpu-offload-gb 2.2`
- `--max-model-len 128000`
- `--kv-cache-dtype fp8`
- `--mamba-cache-dtype float16`
- `--mamba-ssm-cache-dtype float16`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 1280`
- `--enforce-eager`
- `--enable-prefix-caching`
- `--enable-auto-tool-choice`
- `--tool-call-parser hermes`
- `--reasoning-parser qwen3`
- `--trust-remote-code`

Env/volumes match the 27B compose (HF cache ro, `vllm_cache_moe` volume,
`HF_TOKEN` from host env, `expandable_segments:True`).

### Verified performance (35B MoE)

`make bench35 TURNS=54` grows the session to seq 127 460 ≈ the 128k ceiling.
Condensed (every 8th turn + final):

```
turn  prompt    out    seq   cached uncached  ttft_s prefill_tps  out_tps  lat_s  hit%
   1    1885   491    2376     1072      813    0.84       966.5     15.3   32.8   57
   8   18330   491   18821    15008     3322    3.57       930.3     15.4   35.4   82
  16   37213   491   37704    34304     2909    2.92       997.5     15.5   34.6   92
  24   56109   491   56600    53600     2509    2.93       856.5     15.4   34.8   96
  32   75005   491   75496    71824     3181    3.64       873.1     15.4   35.5   96
  40   93901   491   94392    91120     2781    3.52       790.2     15.5   35.2   97
  48  112797   491  113288   110416     2381    3.41       698.6     15.4   35.2   98
  54  126969   491  127460   124352     2617    3.75       697.9     15.4   35.6   98
```

- **Output TPS flat at 15.4** across the full 128k — even flatter than the 27B
  (which dropped 21.9 → 19.6), because the MoE has fewer full-attn layers (10 vs
  16) and a smaller hidden (2048 vs 5120) → less context cost per token.
- **Prefill TPS 966 → 698** (−28 %) — a gentler slope than the 27B's −49 %,
  again because fewer/smaller full-attn layers attend over the prefix.
- **Latency flat ~35 s** end-to-end (decode dominates: `491/15.4 = 31.9 s` of the
  35.6 s final turn); no growth across 128k.
- **Prefix-cache hit climbs 57 → 98 %** (cached = 124 352 = 116 × 1072-token
  blocks at turn 54); only the new ~2.6k-token turn is prefilled.

### Offload efficiency cost

The 2.2 GiB UVA offload is not free at serving time:

- **One CPU thread sits at ~100 %** — vLLM v1's `EngineCore.run_busy_loop`
  busy-polls the scheduler queue for low latency (~73 % even at zero offload) and
  the UVA offloader orchestration pushes it to 100 % under load. This is a design
  choice, not a bug; it is present at 0 offload.
- **GPU power caps at ~80 %** (120 W / 150 W) — the GPU is starved ~20 % of the
  step by PCIe stalls reading the offloaded weight fraction + `--enforce-eager`
  kernel-launch bubbles + batch=1 (no overlap). 80 % is the efficiency
  ceiling of the offload path: decode would be higher on a GPU-only deployment
  (see [Two A10 cards](#two-a10-cards--tp2-projections-for-dense-and-moe)).
- **vLLM-log "2.4 t/s" and "0 % hit" are misleading at idle** — the Avg
  generation throughput is a rolling average diluted by idle gaps, and a single
  idle sample shows 0 % hit. The per-turn bench numbers are authoritative, not the idle log.

### MoE checkpoint status

Only the QuantTrio checkpoint is servable; the other two AWQ builds of this model
are dead ends — do not retry without reason.

| checkpoint | group | lang-only size | status |
|---|---|---|---|
| **QuantTrio/Qwen3.6-35B-A3B-AWQ** | 128 | ~21.5 GiB | **SERVABLE** per the envelope above (Marlin WNA16 MoE, the only viable MoE backend on Ampere) |
| mattbucci/Qwen3.6-35B-A3B-AWQ | 128 | 19.05 GiB | **GARBAGE** — hybrid FP16+AWQ linear-attn layout breaks stacked-shard fusion on both vLLM v0.24.0 (180 skips) and nightly (`MergedColumnParallelLinear has no attribute 'data'`); SGLang's `in_proj_ba` fusion also fails. Fits without offload but unusable regardless. |
| cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit | 32 | ~22.4 GiB | **OOM at construction** — only routed experts AWQ; linear_attn + self_attn + shared_expert + embeddings + lm_head all FP16; group-32 scales inflate. 22.4 > 22.06, param-bound (independent of context/offload). |

FP8 weights are the wrong lever on A10 (Ampere sm86 has no native FP8 tensor
cores): FP8 = 1 byte/param = 35 GiB (2× INT4), worse for fit. FP8 is a Hopper
feature. (FP8 *KV* via FlashInfer works on Ampere and is used here.)

---

## Prefix caching

vLLM supports prefix caching for these hybrid models **experimentally**.
`--enable-prefix-caching` forces `mamba_cache_mode='align'` and an attention
block size set by the mamba page alignment:

| model | align block size |
|---|---|
| 27B Dense | **800 tokens** |
| 35B MoE | **1072 tokens** |

- **Long prompts hit well**: 98.8 % hit on a 2430-token prompt (27B, 3×
  identical). At 18k context, prefix caching cuts TTFT from ~21.6 s (full
  re-prefill) to ~2.96 s (~7×). The 35B MoE hits 98 % by 112k context.
- **Short prompts (< one block) get 0 % hit** — align mode matches only complete
  blocks; a short prompt is one partial block with nothing to match. Coarser
  than ollama's small-segment cache, which is why tiny prompts hit on ollama but
  not here. For real chat (system prompt + history >> block size) it
  matches/beats ollama.
- **Measure via `/metrics` at the root** (`http://host:8000/metrics`), not
  `/v1/metrics`. Counters: `vllm:prefix_cache_queries_total`,
  `vllm:prefix_cache_hits_total` (cumulative token counts). v0.24.0 returns
  `usage.prompt_tokens_details: null`, so usage can't be used for hit rate.

### The reported `hit_rate` lags the real per-turn hit

vLLM v1's `CachingMetrics.hit_rate` (stats.py:107) is
`aggregated_query_hit / aggregated_query_total` over a **rolling deque of the
past N requests** — it is a lagging moving average, not the current request's
hit rate. During the 35B bench the *reported* rate reads ~36 % while the
per-turn hit (from `/metrics` deltas) is already ~90 %+. By the end of the run
the reported rate converges to ~98 %. A low reported `hit_rate` mid-run is not a
bug — the per-turn bench column is authoritative, or wait for the rolling window to fill.

## PCIe bandwidth — grounding the TP=2-on-x4 question

`make bench_pcie` (`scripts/pcie_bw_bench.py`) measures GPU↔host cudaMemcpy
bandwidth and bounds the TP=2 all-reduce cost on the x4 link. A10 negotiates
**Gen4 x4**; measured steady-state (large payloads, pinned + non_blocking):

```
H2D ~6.65 GB/s    D2H ~6.59 GB/s    (~84% of Gen4 x4 line rate ~7.9 GB/s)
```

A 1-GPU test bounds a 2-GPU TP=2 conclusion: NCCL's all-reduce uses **P2P** (best
case) or **SHM fallback** (worst case, if P2P is blocked by ACS/topology). The
SHM per-hop cost is exactly a cudaMemcpy D2H/H2D — measurable here. P2P does one
link hop (GPU0→GPU1); SHM does two (D2H + H2D). So SHM is the conservative upper
bound and P2P is ~half of it.

TP=2 all-reduce volume (model geometry: hidden=5120, layers=64, fp16). Per-layer
tensor = 10 KiB at decode batch=1; one-way = 64 × 10 KiB = 640 KiB/token (decode),
1.31 GB for a 2000-token prefill.

| | SHM fallback (worst) | P2P (≈half) |
|---|---|---|
| decode (47 ms step) | **0.43 %** (203 µs) | ~0.2 % |
| prefill 2000 (2.0 s) | **19.8 %** (396 ms) | ~10 % |

**Conclusion:** on x4, TP=2 all-reduce is negligible for decode (<0.5 %) — the
>20-tps goal is unhurt by comm; GDDR6-read parallelism (each GPU reads half the
weights) is what gives ~2× decode. Prefill pays a real ~10–20 % comm tax on x4
(1026 → ~850–870 prefill tps). The decisive unknown this 1-GPU test can't
resolve is whether NCCL selects P2P or SHM on the 2-GPU board; identifiable only from the
`NCCL_DEBUG=INFO` log.

> Caveat: PCIe ASPM downshifts the link to Gen1 at idle; `nvidia-smi` at rest
> reports Gen1. The bench warms the link first so the reported gen matches the
> achieved bandwidth (Gen4).

## Two A10 cards — TP=2 projections for Dense and MoE

Two A10s on a single host (no NVLink — x4 only): lifts context ceilings and removes
the MoE offload tax. The PCIe grounding above shows **TP=2 pays off for decode**
(all-reduce <0.5 % of the step); prefill pays ~10–20 %.

```
                       single host — A10s have NO NVLink
  ┌──────────────────────────────────────────────────────────────────────┐
  │                            CPU + host RAM                            │
  │              /dev/shm  (NCCL SHM-fallback path)                      │
  └────────┬─────────────────────────────────────┬───────────────────────┘
           │ PCIe Gen4 x4                        │ PCIe Gen4 x4
           │ ~6.6 GB/s each way                  │
  ┌────────▼───────────────────────┐    ┌────────▼───────────────────────┐
  │          A10 #0                │    │           A10 #1               │
  │  24GB GDDR6 @ ~600 GB/s        │    │  24GB GDDR6 @ ~600 GB/s        │
  │                                │    │                                │
  │  TP=2 shard per layer:         │    │  TP=2 shard per layer:         │
  │   • half the weights           │    │   • half the weights           │
  │   • half KV + mamba state      │    │   • half KV + mamba state      │
  └────────┬───────────────────────┘    └────────┬───────────────────────┘
           │                                     │
           └───────────── all-reduce ────────────┘
                 per layer, over x4:
                   decode   ~640 KiB/token  → <0.5% of step
                   prefill  ~1.31 GB / 2000 → ~10–20% of step
                 transport: P2P (1 hop) if ACS allows, else SHM (2 hops)
```

### Footprint: MTP + visual are downloaded, not loaded

Both Qwen3.6 checkpoints ship with a **visual branch** and an **MTP head**.
Both inflate the **download**, not the **VRAM footprint** — they are cut at load:

| component | on disk | in VRAM | cut by |
|---|---|---|---|
| visual branch | ~1–1.5 GiB | no | `--language-model-only` prefix-filters visual weights |
| MTP head | ~0.1–0.34 GiB | no | default `speculative_config=None` → `skip_prefixes=["mtp."]` (opt-in via `--speculative-config`) |
| language model | yes | **yes** | — |

**VRAM goes to the language-model weights + KV/state caches only.** Caveat:
`--language-model-only` skips *loading* visual weights but vLLM still *constructs*
the visual module, so a checkpoint that **quantized** the visual branch asserts at
construction (the mattbucci-CT trap — see [MoE checkpoint status](#moe-checkpoint-status)).
MTP, if opted in, OOMs on a 24GB card (drafter head ~340 MiB, <220 MiB free) —
needs a ≥40 GB GPU.

### Why TP=2, not PP=2, for single-user serving on x4

- **TP=2 parallelizes the GDDR6 weight reads** (the actual decode bottleneck) —
  each GPU reads half the weights per step → ~2× decode tps. All-reduce at
  batch=1 is ~1.3 MB/token, negligible on x4.
- **PP=2 at batch=1 does not overlap** (no pipeline to fill) — stages run
  serially, one GPU idle each half-step → ~same latency as a single GPU that
  could fit the model, *plus* comm. PP makes a too-big model **fit** but does
  not make it **faster**. PP's smaller comm footprint only wins at **high
  batch / multi-user throughput**, a different goal.

### Projected decode tps

Dense decode is GDDR6-bound: `tps ≈ BW / weights_read_per_token × ~0.7`. The formula
is decode-only — the MoE 2× decode is a heuristic (see the bullet below) and 2×
prefill is **not modeled**. 1×A10 rows are measured; 2×A10 rows are untested
projections (no second GPU available for measurement).

| model | setup | offload | decode tps | prefill tps | context | fits 2×A10? |
|---|---|---|---|---|---|---|
| 27B Dense | 1× A10 | none | ~21 | ~1026 | 64k | n/a (measured) |
| 27B Dense | 2× A10 TP=2 | none | ~40–42 | not modeled | 64k+ | yes (overkill) |
| 35B MoE | 1× A10 | **2.2 GiB** | ~15.4 | ~966→698 | 128k | n/a (measured) |
| 35B MoE | 2× A10 **TP=2** | **none** | **~20–24 (proj)** | not modeled | **256k (proj)** | **yes (proj)** |

The 2×A10 MoE projection's reasoning:

- **No offload needed** — ~21.5 GiB / 2 ≈ 11 GiB/GPU weights, ample room for
  KV+state. Removing offload removes the PCIe-stall tax (the 80 %-power ceiling)
  and the 100 %-CPU offloader thread.
- **Decode rises from 15.4 to ~20–24 tps — a heuristic, not a formula output.**
  The dense `BW / weights_read × 0.7` model does **not** transfer to a sparse
  MoE: plugging the active read into it gives an absurd result either way —
  ~260–280 tps at the 1× total (~1.4–1.5 GiB/token), ~560 tps at the TP=2
  per-GPU read (~0.7 GiB). The active read is too
  small to be memory-bandwidth-bound at batch=1 — 1× MoE reads only ~1.5
  GiB/token yet manages 15.4 tps (implies ~23 GiB/s, ~4 % of the A10's ~600
  GB/s GDDR6), so decode is **stall-/overhead-bound** (UVA PCIe stalls +
  `--enforce-eager` kernel-launch bubbles + MoE routing). The lift has two
  sources: **(1)** removing the offload kills the ~20 % PCIe-power stall (the
  80 %-power ceiling) — the dominant effect, moving the floor toward ~19–20
  tps; **(2)** TP=2 halves the always-active backbone read (q/k/v, linear-attn
  in_proj, shared expert, embed, lm_head — read every token regardless of
  routing), a modest extra lift toward ~24. This is far short of the dense
  model's ~2× precisely because the step is stall/overhead-bound, not
  memory-bandwidth-bound; subject to verification on real hardware.
- **Context to 256k (conservative — VRAM is not the limiter)** — ~10 GiB/GPU
  for KV+state is far more than 256k needs: scaling the measured 1× ratio
  (1.78 GiB → 150 349 tokens) by the halved per-GPU KV gives ~1.7 M tokens of
  capacity per GPU. 256k is therefore bounded by the model's `--max-model-len` /
  max-context config, not by VRAM; the 1072-token mamba align block still
  requires `--max-num-batched-tokens ≥ 1072`.

### Config diff (single-GPU → 2-GPU TP=2)

| setting | 27B single A10 | 27B 2× TP=2 | 35B MoE single A10 | 35B MoE 2× TP=2 |
|---|---|---|---|---|
| `--tensor-parallel-size` | 1 | 2 | 1 | 2 |
| `gpus` | all (1) | all (2) | all (1) | all (2) |
| `--gpu-memory-utilization` | 0.97 | 0.97 | 0.95 | 0.95–0.97 (more headroom) |
| `--cpu-offload-gb` | — | — | **2.2** | **drop (no offload needed)** |
| `--max-model-len` | 64000 | 64000+ | 128000 | **256000 (proj)** |
| `--max-num-batched-tokens` | 1024 | 1024 | 1280 | 1280 (≥1072 align) |
| `--max-num-seqs` | 1 | 1 | 1 | 1 |
| weights / GPU | 18.83 GiB | ~9.4 GiB | ~19.2 GiB (+2.2 off) | ~10.75 GiB |
| `shm_size` | 32g | 32g | 32g | 32g |
| `NCCL_DEBUG` env | — | `INFO` | `INFO` | `INFO` |

### How to verify when running it for real

- `nvidia-smi` per-GPU memory **balanced** (~equal) → TP shards both layer types.
- `NCCL_DEBUG=INFO` log: `via P2P/IPC` (best) or `via SHM` (fallback, fine);
  `via NET` would indicate TCP (shouldn't happen intra-node).
- 27B decode ~40–42 tps, 35B MoE decode > 15.4 (no offload stall) → TP working;
  ~21/15.4 → not sharding the hybrid layers; init error → TP unsupported for the
  hybrid arch.

### Open risks (not resolvable on a 1-GPU box)

- **vLLM's TP sharding of the GatedDeltaNet linear-attn layers is unverified**
  — the main risk for both models. If TP only shards the full-attn layers, load
  is imbalanced and the 2× never materializes. The balanced-memory check above
  catches this.
- x4 prefill tax ~10–20 % (P2P vs SHM).
- PP=2 at batch=1 → ~same latency as 1 GPU + comm; don't use PP for single-user.

## Experiments that did NOT work (do not retry without reason)

### MTP speculative decoding — does not fit on this GPU (27B)

The checkpoint has an MTP head (`mtp_num_hidden_layers = 1`). vLLM v0.24.0 ships
`vllm/model_executor/models/qwen3_5_mtp.py` and registers `Qwen3_5MTP`; enable
with `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'`.

It OOMs at **drafter weight-load**, before any speculation runs, at every util
tried (0.97, 0.94, 0.91) and `--max-num-batched-tokens` 1024 and 256:

```
Loading weights took 5.2s        # main model 18.83 GiB
Loading drafter model...          # OOM: tried to allocate 340.00 MiB, 224 MiB free
```

Root cause: at drafter load the GPU holds weights (18.83 GiB) + fixed overhead
(~3 GiB = FP16 embedding ~1.56 + CUDA ctx ~0.6 + cuDNN workspaces ~0.5) ≈ 21.8
GiB, leaving only ~184–224 MiB. The MTP head needs ~340 MiB. The shortfall is
**independent of `num_speculative_tokens`** (1 vs 2 changes only runtime draft
activations, not the head footprint) and **independent of util** (drafter loads
before KV sizing). `--max-num-batched-tokens` 1024→256 freed only ~40 MiB — the
overhead is fixed, not batch-token-driven.

**Conclusion:** MTP is infeasible on this 24GB A10 with this 18.83 GiB model; a
≥40 GB GPU would fit it. Expected gain if it fit: ~1.3–1.7× output TPS
(single-token speculation, 1-layer head, ~50–70 % acceptance) → ~28–35 tps, not 2×.

### CUDA graphs (removing `--enforce-eager`) — costs more context than it gains

Removing `--enforce-eager` enables `torch.compile` + CUDA graph capture. Compile
worked (`cudagraph_capture_sizes: [1, 2]`), but engine init then failed KV sizing:

```
ValueError: 2.1 GiB KV cache is needed, which is larger than the available KV
cache memory (1.86 GiB). Estimated maximum model length is 56000.
```

CUDA-graph memory profiling reserves memory for captured graphs, so
`--gpu-memory-utilization=0.97` behaves like 0.9514 — KV drops ~2.1 → ~1.86 GiB,
capping context at ~56k. Keeping 64k would need util ~0.99 (which OOMs). So CUDA
graphs trade ~8k of context (64k → 56k) for a modest kernel-launch saving — not
worth it, and the hybrid GatedDeltaNet layers add compile risk. `--enforce-eager`
stays.

### CPU offload on the 27B (6 GiB) — rejected as too slow

`--cpu-offload-gb 6` gives 128k context easily on the dense 27B but decode
drops to ~2 tps (PCIe weight reads every step — the *whole* 27B reads CPU each
token because every layer is active). Rejected as too slow. This is the opposite
of the MoE path: on the MoE, only the offloaded expert fraction reads CPU and
routing often misses the offloaded experts, so 2.2 GiB costs ~15 tps, not 2.

### Dead MoE checkpoints

See [MoE checkpoint status](#moe-checkpoint-status) — mattbucci (garbage,
hybrid-quant fusion breaks both runtimes) and cyankiwi (OOM at construction,
param-bound). Do not retry without reason.

## Hardware ceiling facts

- **120k no-offload is impossible on this A10 for the dense 27B**: 22.06 −
  18.83 = 3.23 GiB free; at the measured ~35 KiB/token, 120k needs ~4.0 GiB —
  over budget. Hard upper bound ~72–80k tokens (FP16 state; the bare per-token
  rate allows ~95k, but non-KV overhead lowers the real ceiling).
- **The MoE 35B is param-bound without offload** — weights 21.38 + reserve 0.44
  = 21.82 GiB > the 21.40 GiB achievable at util 0.97. No amount of context fits
  without `--cpu-offload-gb`; 2.2 GiB offload lifts it to 128k.
- **`--gpu-memory-utilization` > 0.98 fails** on this A10: free VRAM
  21.83/22.06 GiB < 0.99×22.06.
- **FP8 KV on Ampere** works via FlashInfer (auto-selected); no native FP8 tensor
  cores needed. `int4_per_token_head` KV is nightly-only; TurboQuant 4-bit KV is
  broken on hybrid Qwen3.5.

## Gotchas

- **open-webui persisted base URL**: `OPENAI_API_BASE_URL` env only seeds a fresh
  DB. The `open-webui-data` volume persists `openai.api_base_urls` in sqlite
  `config`, which takes precedence → empty model dropdown if it still points at
  an old backend. Fix: update that DB row to `http://vllm:8000/v1` and restart,
  or wipe the volume.
- **Thinking mode**: the model is a thinking model. Without `--reasoning-parser
  qwen3`, thinking tags are stripped (special tokens) and raw CoT dumps into
  `content`, eating `max_tokens` → truncated answer. The parser moves thinking
  to `message.reasoning`. Disable thinking per-request with
  `chat_template_kwargs: {"enable_thinking": false}` (no server-side CLI flag
  for this in v0.24.0).
- **`HF_TOKEN`**: resolves from host env (`${HF_TOKEN}`); not committed. The
  mounted HF cache is read-only and offline (`HF_HUB_OFFLINE=1`).
- **Idle vLLM log lines are misleading**: `Avg generation throughput: 2.4 t/s`
  and `Prefix cache hit rate: 0.0%` at idle are a rolling-average artifact and
  an empty-window sample — not the serving rate. The per-turn bench column is authoritative.
