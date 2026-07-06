# Qwen3.6-27B-AWQ on a single 23 GB A10 — grounded vLLM deployment notes

Serves the **27B hybrid model at full 64k context with no CPU offload** on one
NVIDIA A10 (23 GB), with prefix caching and **~21 output tps that stays flat to
max context**. This repo records what works, what doesn't, and the measurements
behind both — for anyone squeezing a big model onto a small GPU.

- vLLM OpenAI API: `http://localhost:8000/v1` (served-model-name `qwen3.6-27b`)
- open-webui: `http://localhost:3000/`

## Why it fits on 23 GB: AWQ + hybrid linear attention + tuned caches

Three things together make 64k context on a 23 GB GPU possible — and keep decode
fast at long context. This is the core of the deployment.

### 1. AWQ INT4 weights on the fast Marlin kernel

`QuantTrio/Qwen3.6-27B-AWQ` is genuinely packed INT4 (not just a name), and vLLM
runs it on **AWQ-Marlin** (fused dequant + INT4 GEMM) — *not* dequantized to FP16.
From the load log: `Using MarlinLinearKernel for AutoAWQMarlinLinearMethod`,
`quantization=auto_awq`, plus fused `norm_quant` / `act_quant` passes.

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
decode would be ~3× slower. Marlin keeps INT4 packed in HBM and dequants inside
the GEMM, so you get the full memory/bandwidth benefit.

### 2. Hybrid architecture — 48 of 64 layers are linear (GatedDeltaNet)

This is **why output tps stays flat as context grows.** Only 16 layers are
full-attention (KV-cached, `O(context)` per token); 48 are linear-attention with
a fixed-size SSM state (`O(1)` per token). 75% of layers don't pay the
growing-context cost.

```
Qwen3.6-27B-AWQ  —  64 transformer layers (hybrid)

  ┌──────────────────────────────────────────────┐
  │  16 × full-attention layers                  │  KV cache (FP8)
  │     self_attn q/k/v  — FP16                   │  O(context) / token
  │     MLP gate/up/down — INT4 AWQ-Marlin        │  ← grows with context
  └──────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────┐
  │  48 × GatedDeltaNet linear-attention layers  │  SSM state cache (FP16)
  │     linear_attn in_proj — FP16               │  O(1) / token
  │     MLP gate/up/down   — INT4 AWQ-Marlin      │  ← fixed cost at any ctx
  └──────────────────────────────────────────────┘

  + 1 MTP head (present, unused — see Experiments)
  + visual branch (skipped via --language-model-only)
```

Measured across a growing session (2k → 64k context), **decode barely moves**:

| context | out_tps (decode) | vs 2k | prefill_tps (uncached new turn) |
|---|---|---|---|
| 2 k | 21.9 | — | 1026 |
| 18 k | 20.9 | −5% | 802 |
| 40 k | 20.2 | −8% | 646 |
| 64 k | 19.6 | **−10%** | 526 |

Decode slows only ~10% over 62 k tokens of context because the 48 linear layers
are `O(1)`. Prefill of the new turn slows ~49% because the 16 full-attention
layers attend over the entire cached prefix per prefilled token. See
[Verified performance](#verified-performance).

### 3. KV + state caches tuned for the hybrid mix

| cache | dtype | covers | why |
|---|---|---|---|
| KV cache | **fp8** | 16 full-attn layers | FlashInfer on Ampere; halves KV HBM |
| mamba state cache | **float16** | 48 linear-attn layers | **key lever** — config defaults FP32; FP16 lifts the no-offload ceiling 54.9k → 72.5k tokens |
| prefix cache | on (align mode) | cross-request reuse | experimental for hybrid; 800-token block size |

Without the `--mamba-ssm-cache-dtype float16` lever, 64k context would not fit
on this GPU. See [Working config](#working-config).

## Hardware

| | |
|---|---|
| GPU | NVIDIA A10, Ampere sm86, 23 GB VRAM (~22.06 GiB usable by vLLM) |
| PCIe | Gen4 **x4** (card is x16-capable; slot/wiring caps at x4) |
| TP | 1 (single GPU) |
| Weights on GPU | 18.83 GiB |
| KV + state cache | ~2.15 GiB → 64,744 tokens |

## Quick start

```
make            # help (default)
make run        # docker compose up          (foreground; Ctrl-C to stop)
make start      # docker compose up -d       (detached)
make stop       # docker compose stop
make bench      # growing coding-session bench to ~64k context (override: TURNS=N)
make bench_pcie # GPU<->host PCIe bandwidth (free GPU needed: make stop first)
```

The coding-session bench (`scripts/coding_session_bench.py`) streams a chat that
grows one ~2k-token user message + ~500-token assistant reply per turn and
reports per-turn prompt/seq/cached/uncached tokens, TTFT, prefill TPS, output
TPS. Cache hit is read from vLLM `/metrics` (at the **root**, not `/v1`).

## Working config

`docker-compose.yaml` vLLM command:

```
--model QuantTrio/Qwen3.6-27B-AWQ --served-model-name qwen3.6-27b
--language-model-only
--tensor-parallel-size 1 --gpu-memory-utilization 0.97 --max-model-len 64000
--kv-cache-dtype fp8 --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16
--max-num-seqs 1 --max-num-batched-tokens 1024
--enforce-eager --enable-prefix-caching
--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
--trust-remote-code
```

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

## Verified performance

`make bench` (default `TURNS=27`) grows the session to seq 63924 ≈ the 64k
ceiling. Condensed (every 8th turn + final):

```
turn  prompt   out     seq  cached uncached  ttft_s prefill_tps  out_tps  lat_s  hit%
   1    1885   474    2359       0     1885    1.84      1026.5     21.9   23.5    0
   8   18369   500   18869   16000     2369    2.96       801.7     20.9   26.8   87
  17   39704   500   40204   36800     2904    4.49       646.2     20.2   29.2   93
  27   63424   500   63924   60800     2624    4.99       525.8     19.6   30.5   96
```

- **Output TPS ~21 → 19.6** — HBM-bandwidth-bound (~600 GB/s, 18.83 GiB weights).
  Stays nearly flat to 64k because 48/64 layers are linear (`O(1)`/token).
- **Prefill TPS 1026 → 526** (uncached new turn) — the 16 full-attn layers
  attend over the growing prefix per prefilled token (`O(context × uncached)`).
- **Latency 23.5 → 30.5 s** (+30% over 62 k tokens); decode is ~84% of latency at
  max context (`500/19.6 = 25.5 s` of the 30.5 s turn). Modest growth because the
  dominant decode term barely moves.
- **Prefix caching holds at long context** — hit 96% at 64k (cached = 60,800 =
  76 × 800-token blocks); only the new turn is prefilled.

> Note on turn 1: a cold start gives 0% hit (ttft ~1.8 s). Re-running with the
> same user text warms the prefix cache and turn 1 hits ~85% (ttft ~0.3 s). The
> table above is a cold start.

## Prefix caching

vLLM v0.24.0 supports prefix caching for this hybrid model **experimentally**.
`--enable-prefix-caching` forces `mamba_cache_mode='align'` and an **attention
block size of 800 tokens**.

- **Long prompts hit well**: 98.8% hit on a 2430-token prompt (3× identical).
  At 18k context, prefix caching cuts TTFT from ~21.6 s (full re-prefill) to
  ~2.96 s (~7×).
- **Short prompts (< 800 tokens) get 0% hit** — align mode matches only complete
  800-token blocks; a short prompt is one partial block with nothing to match.
  Coarser than ollama's small-segment cache, which is why tiny prompts hit on
  ollama but not here. For real chat (system prompt + history >> 800 tokens) it
  matches/beats ollama.
- **Measure via `/metrics` at the root** (`http://host:8000/metrics`), not
  `/v1/metrics`. Counters: `vllm:prefix_cache_queries_total`,
  `vllm:prefix_cache_hits_total` (cumulative token counts). v0.24.0 returns
  `usage.prompt_tokens_details: null`, so usage can't be used for hit rate.

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
1.31 GB for a 2k-token prefill.

| | SHM fallback (worst) | P2P (≈half) |
|---|---|---|
| decode (47 ms step) | **0.43%** (203 µs) | ~0.2% |
| prefill 2k (2.0 s) | **19.8%** (396 ms) | ~10% |

**Conclusion:** on x4, TP=2 all-reduce is negligible for decode (<0.5%) — the
>20-tps goal is unhurt by comm; HBM-read parallelism (each GPU reads half the
weights) is what gives ~2× decode. Prefill pays a real ~10–20% comm tax on x4
(1026 → ~850–870 prefill tps). The decisive unknown this 1-GPU test can't
resolve is whether NCCL selects P2P or SHM on the 2-GPU board — confirm with
`NCCL_DEBUG=INFO`.

> Caveat: PCIe ASPM downshifts the link to Gen1 at idle; `nvidia-smi` at rest
> reports Gen1. The bench warms the link first so the reported gen matches the
> achieved bandwidth (Gen4).

## Two A10 cards — proposed TP=2 for a 1.5× bigger model

Goal: serve a ~40B (1.5×) model at >20 output tps, single-user. The x4 link has
no NVLink between A10s, so the question is whether TP=2 still pays off. The PCIe
grounding above says **yes for decode** (all-reduce <0.5% of the step); prefill
pays ~10–20%.

```
                       single host — A10s have NO NVLink
  ┌─────────────────────────────────────────────────────────────┐
  │                       CPU + host RAM                         │
  │              /dev/shm  (NCCL SHM-fallback path)              │
  └─────────┬──────────────────────────────────────────┬─────────┘
            │ PCIe Gen4 x4                               │ PCIe Gen4 x4
            │ ~6.6 GB/s each way                          │
  ┌─────────▼─────────────────┐    ┌──────────────────────▼─────────┐
  │          A10 #0           │    │           A10 #1                │
  │  23 GB HBM @ ~600 GB/s    │    │  23 GB HBM @ ~600 GB/s          │
  │                           │    │                                 │
  │  TP=2 shard per layer:    │    │  TP=2 shard per layer:          │
  │   • ~9.4 GiB weights      │    │   • ~9.4 GiB weights            │
  │   • half KV + mamba state │    │   • half KV + mamba state       │
  └─────────┬─────────────────┘    └──────────────────────┬─────────┘
            │                                              │
            └────────────── all-reduce ────────────────────┘
                 per layer, over x4:
                   decode   ~640 KiB/token  → <0.5% of step
                   prefill  ~1.31 GB / 2k   → ~10–20% of step
                 transport: P2P (1 hop) if ACS allows, else SHM (2 hops)
```

Why **TP=2, not PP=2**, for single-user serving on x4:

- **TP=2 parallelizes the HBM weight reads** (the actual decode bottleneck) —
  each GPU reads half the weights per step → ~2× decode tps. All-reduce at
  batch=1 is ~1.3 MB/token, negligible on x4.
- **PP=2 at batch=1 does not overlap** (no pipeline to fill) — stages run
  serially, one GPU idle each half-step → ~same latency as a single GPU that
  could fit the model, *plus* comm. PP makes a too-big model **fit** but does
  not make it **faster**. PP's smaller comm footprint only wins at **high
  batch / multi-user throughput**, a different goal.

Expected decode tps (HBM-bound: `tps ≈ BW / weights_read_per_token × ~0.7`):

| model | setup | decode tps | prefill tps | fits 2×A10? |
|---|---|---|---|---|
| 27B AWQ (18.83 GiB) | 1× A10 (this repo) | ~21 | ~850–1026 | n/a |
| 27B AWQ | 2× A10 TP=2 | ~35–40 | ~1600–1700 | yes (overkill) |
| 40B AWQ (~28 GiB, 1.5×) | 1× A10 | — | — | no |
| 40B AWQ | 2× A10 **TP=2** | **~25–30** | ~800–850 | **yes ✓** |
| 40B AWQ | 2× A10 PP=2 | ~14–16 | ~680 | yes (slower) |

Config diff (single-GPU → 2-GPU TP=2):

| setting | single A10 (current) | 2× A10 TP=2 |
|---|---|---|
| `--tensor-parallel-size` | 1 | 2 |
| `gpus` | all (1) | all (2) |
| `--max-num-seqs` | 1 | 1 (single-user) |
| `--gpu-memory-utilization` | 0.97 | 0.97 |
| `--max-model-len` | 64000 | 64000 (more headroom; keep) |
| `ipc: host` | drop | drop (private `/dev/shm` via `shm_size` is enough for one container) |
| `shm_size` | 32g | 32g (intra-container ranks share it) |
| `NCCL_DEBUG` env | — | `INFO` (verify P2P vs SHM) |
| weights / GPU | 18.83 GiB | ~9.4 GiB (27B) / ~14 GiB (40B) |

How to verify when running it for real:

- `nvidia-smi` per-GPU memory **balanced** (~equal) → TP shards both layer types.
- `NCCL_DEBUG=INFO` log: `via P2P/IPC` (best) or `via SHM` (fallback, fine);
  avoid `via NET` (would mean TCP, shouldn't happen intra-node).
- Decode tps ~35–40 (27B) → TP working; ~21 → not sharding the hybrid layers;
  init error → TP unsupported for this hybrid arch.

Open risks (not resolvable on a 1-GPU box):

- **vLLM's TP sharding of the 48 GatedDeltaNet linear-attn layers is unverified**
  — the main risk. If TP only shards the 16 full-attn layers, load is imbalanced
  and the 2× never materializes. The balanced-memory check above catches this.
- x4 prefill tax ~10–20% (P2P vs SHM).
- PP=2 at batch=1 → ~15 tps; don't use PP for single-user serving.

## Experiments that did NOT work (do not retry without reason)

### MTP speculative decoding — does not fit on this GPU

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

**Conclusion:** MTP is infeasible on this 23 GB A10 with this 18.83 GiB model; a
≥40 GB GPU would fit it. Expected gain if it fit: ~1.3–1.7× output TPS
(single-token speculation, 1-layer head, ~50–70% acceptance) → ~28–35 tps, not 2×.

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

### CPU offload — rejected earlier

`--cpu-offload-gb 6` gives 128k context easily but decode drops to ~2 tps (PCIe
weight reads every step). Rejected as too slow. The no-offload path above is the
working base.

## Hardware ceiling facts

- **120k no-offload is impossible on this A10**: 22.06 − 18.83 = 3.23 GiB max
  cache; 120k needs ~3.55 GiB. Hard upper bound ~72–80k tokens (FP16 state cache).
- **`--gpu-memory-utilization` > 0.98 fails**: free VRAM 21.83/22.06 GiB <
  0.99×22.06.
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

## Repo layout

| file | what |
|---|---|
| `docker-compose.yaml` | vLLM + open-webui stack |
| `Makefile` | `run` / `start` / `stop` / `bench` / `bench_pcie` |
| `scripts/coding_session_bench.py` | growing coding-session bench (prefill/output tps, cache hit from `/metrics`) |
| `scripts/pcie_bw_bench.py` | GPU↔host PCIe D2H/H2D bandwidth + TP=2 all-reduce estimate |
| `README.md` | this doc |