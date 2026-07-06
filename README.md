# openweb — Qwen3.6-27B-AWQ on vLLM behind open-webui

Serves `QuantTrio/Qwen3.6-27B-AWQ` via vLLM v0.24.0 on a single NVIDIA A10 (23 GB),
fronted by open-webui. Model is a hybrid (16 full-attention + 48 GatedDeltaNet
linear-attention) AWQ INT4 checkpoint, served text-only.

- vLLM OpenAI API: `http://mini4:8000/v1` (served-model-name `qwen3.6-27b`)
- open-webui: `http://mini4:3000/` (login `dmitri` / `1`)

## Run

```
make            # help (default)
make run        # docker compose up  (foreground; Ctrl-C to stop)
make start      # docker compose up -d  (detached)
make stop       # docker compose stop
make bench      # growing coding-session bench to ~64k context (override: TURNS=N)
make bench_pcie # GPU<->host PCIe bandwidth (free GPU needed: make stop first)
```

The bench (`scripts/coding_session_bench.py`) streams a chat that grows one
~2k-token user message + ~500-token assistant reply per turn and reports
per-turn prompt/seq/cached/uncached tokens, TTFT, prefill TPS, output TPS.
Cache hit is read from vLLM `/metrics` (at the **root**, not `/v1`).

## Working config (the committed base)

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

Why each non-obvious flag:

- `--language-model-only` — serve the text branch of the VLM checkpoint.
- `--mamba-ssm-cache-dtype float16` — **key lever**. The config defaults the 48
  linear-attention state caches to FP32; FP16 lifted the no-offload ceiling from
  ~54.9k → ~72.5k tokens. `--kv-cache-dtype fp8` only quantizes the 16 full-attention
  layers' KV.
- `--max-model-len 64000` — stable ceiling. 65536 was stable without prefix caching
  but align-mode overhead (see below) reduces available KV, so 64000 leaves margin.
- `--gpu-memory-utilization 0.97` — 0.98 + 70k context OOM'd on prefill (zero
  activation headroom). 0.97 + `--max-num-batched-tokens 1024` is stable.
- `--enforce-eager` — saves memory (no CUDA graph capture). See "Experiments" for
  why removing it does not pay off here.
- `--enable-prefix-caching` — experimental for this hybrid model; forces
  `mamba_cache_mode='align'` and attention block size = 800 tokens.
- `--reasoning-parser qwen3` — separates thinking into `message.reasoning` so it
  does not eat `max_tokens` and truncate the answer. Disable thinking per-request
  with `chat_template_kwargs: {"enable_thinking": false}`.

Weights ~18.83 GiB on GPU, KV ~2.15 GiB → 64,744 tokens. Env
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. HF cache mounted read-only;
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. `HF_TOKEN` resolves from host env.

## Verified performance (the committed base)

8-turn growing coding session (~2k new input + ~500 output/turn, thinking off):

```
turn  prompt     seq  cached uncached  ttft_s prefill_tps  out_tps  lat_s  hit%
   1    1885    2385    1600      285    0.33       858.3     22.0   23.1   85
   2    4241    4741    4000      241    0.31       784.0     21.7   23.3   94
   3    6599    7099    4000     2599    2.74       948.7     21.5   26.0   61
   4    8955    9455    6400     2555    2.79       917.3     21.4   26.1   71
   5   11311   11811    8800     2511    2.84       884.2     21.2   26.3   78
   6   13667   14167   11200     2467    2.91       847.2     21.1   26.5   82
   7   16023   16523   13600     2423    2.96       819.2     21.0   26.7   85
   8   18379   18879   16000     2379    2.96       804.7     20.9   26.8   87
```

- **Output TPS steady ~21** — decode is HBM-bandwidth-bound on the A10
  (~600 GB/s, 18.83 GiB weights). This is the practical ceiling for this model
  on this GPU.
- **Prefill TPS ~850** (uncached tokens / TTFT).
- **Prefix caching works**: per-turn uncached prefill stays bounded to ~2.4–2.6k
  tokens (the new turn) even as context grows 1.9k → 18.4k. Hit rate climbs to 87%.
  At 18k context, prefix caching cuts TTFT from ~21.6s (full re-prefill) to ~2.96s
  (~7×). The `cached` column steps by 800 (block size); turn N's completed blocks
  become turn N+1's hits.
- **Short prompts (< 800 tokens) get 0% hit** — align mode forces a 800-token
  attention block; prefix caching only matches complete blocks, and a short
  prompt is one partial block with nothing to match. This is coarser than ollama's
  small-segment cache, which is why tiny prompts hit on ollama but not here.

### Long context — speed at full 64k context (27-turn run)

`make bench` grows the session to seq 63924 ≈ the 64k ceiling.
Condensed (every 8th turn + the final turn):

```
turn  prompt   out     seq  cached uncached  ttft_s prefill_tps  out_tps  lat_s  hit%
   1    1885   474    2359       0     1885    1.84      1026.5     21.9   23.5    0
   8   18369   500   18869   16000     2369    2.96       801.7     20.9   26.8   87
  17   39704   500   40204   36800     2904    4.49       646.2     20.2   29.2   93
  27   63424   500   63924   60800     2624    4.99       525.8     19.6   30.5   96
```

`lat_s = TTFT + out/out_tps`. At turn 27: `4.99 + 500/19.6 = 30.5s`, so **decode
is ~84% of latency** even at max context. Latency grows 23.5 → 30.5s (+30%) across
2k → 64k context — modest, because the dominant decode term barely moves.

Two distinct degradation curves:

- **Decode (`out_tps`) 21.9 → 19.6, only ~10% slower at 64k.** Per decode step the
  16 full-attention layers attend over the growing KV (O(context)/token), but the
  48 linear-attention (GatedDeltaNet) layers are O(1)/step (fixed state). 75% of
  layers are cheap → decode stays near the HBM-bandwidth ceiling. This is the
  hybrid architecture paying off at long context.
- **Prefill (`prefill_tps`) 1026 → 526, ~49% slower at 64k.** Prefilling the
  ~2.6k uncached new-turn tokens, those same 16 full-attention layers attend over
  the *entire 63k cached prefix* per new token → O(context × uncached) work. So
  prefix caching keeps the *work* bounded to the new turn (~2.6k tokens) even at
  64k, but the *rate* on that work halves as context grows.
- **Prefix caching holds at long context**: hit 96% at 64k (cached = 60,800 =
  76 × 800-token blocks); only the new turn is prefilled.

Net: the model stays usable at full context. Decode (~19.6 tps, was 21.9) drives
~25.5s of the 30.5s turn; prefill of the new turn adds ~5s TTFT (was 1.8s).

## PCIe bandwidth — grounding the TP=2-on-x4 question

`make bench_pcie` (`scripts/pcie_bw_bench.py`) measures GPU↔host cudaMemcpy
bandwidth on this box and bounds the TP=2 all-reduce cost on the x4 link. The
A10 negotiates **Gen4 x4** (max x16; the slot/wiring caps it at x4). Measured
steady-state (large payloads, pinned + non_blocking):

```
H2D ~6.65 GB/s   D2H ~6.59 GB/s   (~84% of Gen4 x4 line rate ~7.9 GB/s)
```

Why a 1-GPU test bounds a 2-GPU TP=2 conclusion: NCCL's all-reduce uses P2P
(best case) or SHM fallback (worst case, if P2P is blocked by ACS/topology).
The SHM fallback's per-hop cost is exactly a cudaMemcpy D2H/H2D — measurable
here. P2P does one link hop (GPU0→GPU1); SHM does two (D2H+H2D). So the SHM
number is the conservative upper bound, and P2P is ~half of it.

TP=2 all-reduce volume (model geometry from `config.json`: hidden=5120,
layers=64, fp16). Per-layer tensor = 10 KiB at decode batch=1; one-way volume
= 64 × 10 KiB = 640 KiB/token (decode), 1.31 GB for a 2k-token prefill.

| | SHM fallback (worst) | P2P (≈half) |
|---|---|---|
| decode (47 ms step) | **0.43%** (203 µs) | ~0.2% |
| prefill 2k (2.0 s) | **19.8%** (396 ms) | ~10% |

**Conclusion:** on x4, TP=2 all-reduce is negligible for decode (<0.5%) — the
>20-tps goal is unhurt by comm; the HBM-read parallelism (each GPU reads half
the weights) is what gives ~2× decode. Prefill pays a real ~10–20% comm tax
on x4 (1026 → ~850–870 prefill tps). The decisive unknown that this test
cannot resolve on a 1-GPU box is whether NCCL selects P2P or SHM on the
2-GPU board — confirm with `NCCL_DEBUG=INFO` when running TP=2 for real.

Caveat: PCIe ASPM downshifts the link to Gen1 when idle; `nvidia-smi` queried
at rest reports Gen1. The script warms the link first so the reported gen
matches the achieved bandwidth (Gen4).

## Experiments that did NOT work (do not retry without reason)

### MTP speculative decoding — does not fit on this GPU

The checkpoint has an MTP head (`mtp_num_hidden_layers = 1`). vLLM v0.24.0 ships
`vllm/model_executor/models/qwen3_5_mtp.py` and registers `Qwen3_5MTP`, so it is
implemented: enable with
`--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'`.

It OOMs at **drafter weight-load**, before any speculation runs, at every util
tried (0.97, 0.94, 0.91) and at `--max-num-batched-tokens` 1024 and 256:

```
Loading weights took 5.2s        # main model 18.83 GiB
Loading drafter model...          # OOM: tried to allocate 340.00 MiB, 224 MiB free
```

Root cause: at drafter load the GPU holds weights (18.83 GiB) + fixed overhead
(~3 GiB = FP16 embedding ~1.56 + CUDA ctx ~0.6 + cuDNN workspaces ~0.5), totalling
~21.8 GiB and leaving only ~184–224 MiB. The MTP head needs ~340 MiB. The shortfall
is ~115 MiB and is **independent of `num_speculative_tokens`** (1 vs 2 changes only
runtime draft activations, not the head's parameter footprint) and independent of
util (the drafter loads before KV sizing, so util is not yet in play).

`--max-num-batched-tokens` 1024→256 freed only ~40 MiB — the overhead is not
batched-token-driven, it is the fixed embedding + CUDA context.

**Conclusion:** MTP is infeasible on this 23 GB A10 with this 18.83 GiB model.
A larger GPU (≥40 GB) would fit it. Expected gain if it fit: ~1.3–1.7× output TPS
(single-token speculation, 1-layer head, ~50–70% acceptance) → ~28–35 tps, not 2×.

### CUDA graphs (removing `--enforce-eager`) — costs more context than it gains

Removing `--enforce-eager` enables `torch.compile` (inductor) + CUDA graph capture.
Compile worked (`Compiling a graph for compile range (1, 1024) takes 15.78s`;
`cudagraph_capture_sizes: [1, 2]`). But the engine then failed KV sizing:

```
ValueError: 2.1 GiB KV cache is needed, which is larger than the available KV
cache memory (1.86 GiB). Estimated maximum model length is 56000.
```

CUDA-graph memory profiling reserves memory for the captured graphs, so
`--gpu-memory-utilization=0.97` behaves like 0.9514 — KV drops from ~2.1 GiB
(64,744 tokens) to ~1.86 GiB, capping effective context at ~56k. To keep 64k you'd
need util ~0.99 (which OOMs on this GPU). So CUDA graphs trade ~8k of context
(64k → 56k) for a modest decode-kernel-launch saving — not worth it here, and the
hybrid GatedDeltaNet layers add compile risk. `--enforce-eager` stays.

### CPU offload — rejected earlier

`--cpu-offload-gb 6` gives 128k context easily but decode drops to ~2 tps
(PCIe weight reads every step). Rejected as too slow. The no-offload path above
is the working base.

## Hardware ceiling facts

- 120k no-offload is impossible on this A10: 22.06 − 18.83 = 3.23 GiB max cache;
  120k needs ~3.55 GiB. Hard upper bound ~72–80k tokens (FP16 state cache).
- `--gpu-memory-utilization` > 0.98 fails: free VRAM 21.83/22.06 GiB < 0.99×22.06.
- FP8 KV on Ampere works via FlashInfer (auto-selected); no native FP8 tensor cores
  needed. `int4_per_token_head` KV is nightly-only; TurboQuant 4-bit KV is broken
  on hybrid Qwen3.5.

## open-webui gotcha

`OPENAI_API_BASE_URL` env only seeds a fresh DB. The `open-webui-data` volume
persists `openai.api_base_urls` in sqlite `config`, which takes precedence →
empty model dropdown if it still points at an old backend. Fix: update that DB
row to `http://vllm:8000/v1` and restart, or wipe the volume.