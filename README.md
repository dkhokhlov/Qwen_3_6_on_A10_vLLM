# Running Qwen3.6 on NVIDIA A10 24GB with vLLM

[![GPU](https://img.shields.io/badge/GPU-NVIDIA_A10_24GB-blue)](https://www.techpowerup.com/gpu-specs/a10-pcie.c3793)
[![Qwen3.6](https://img.shields.io/badge/Qwen3.6-27B_Dense-success)](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ)
[![Qwen3.6](https://img.shields.io/badge/Qwen3.6-35B_MoE-success)](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)
[![context](https://img.shields.io/badge/context-64k_and_128k-brightgreen)](#dense-vs-moe--why-the-bigger-model-serves-more-context)
[![quant](https://img.shields.io/badge/quant-AWQ_INT4-orange)](#awq-quantization--the-method-and-why-both-models-use-it)
[![license](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)

Serve the latest [**Qwen3.6**](https://huggingface.co/collections/Qwen/qwen36) — a
**dense** model and a **mixture-of-experts (MoE)** model, built for **agentic coding**
and tool use — on a single [NVIDIA A10](https://www.techpowerup.com/gpu-specs/a10-pcie.c3793)
(24GB) with [vLLM](https://github.com/vllm-project/vllm). Qwen reports 73–77% on [SWE-bench](https://www.swebench.com) Verified for the two base models.
Real measured throughput at full context:

- **[Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ) — a
  dense model** (27B active params, no experts) at full **64k context, no CPU
  offload**, output `21tps` flat to max context.
- **[Qwen3.6-35B-A3B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)
  — a MoE model** (35B total / 3B active, 256 routed experts) at full **128k
  context with 2.2 GiB UVA CPU offload**, output `15.4tps` flat to max context.

**The bigger MoE serves *more* context than the smaller dense model** — 128k vs
64k — because MoE has a smaller KV per token (fewer full-attn layers, smaller
hidden dim). The MoE's 35B weights don't fit without offload; the dense 27B fits
cleanly. See [Dense vs MoE — why the bigger model serves more context](#dense-vs-moe--why-the-bigger-model-serves-more-context).

**Both models are [AWQ](https://arxiv.org/abs/2306.00978)-INT4** — activation-aware 4-bit quantization (group
128, asymmetric), not naive round-to-nearest — which is why a 27B and a 35B
model both fit on 24GB at near-FP16 quality. vLLM's [Marlin](https://arxiv.org/abs/2401.00755) kernel keeps them
packed-INT4 through serving, ~3–4× faster to decode than FP16. See
[AWQ quantization — the method, and why both models use it](#awq-quantization--the-method-and-why-both-models-use-it).

## TOC

**Part 1 — Getting Started**
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Repo structure](#repo-structure)
- [Security](#security)

**Part 2 — Single-A10 Deployments**
- [TL;DR — one A10, both models](#tldr--one-a10-both-models)
- [Hardware](#hardware)
- [Dense vs MoE — why the bigger model serves more context](#dense-vs-moe--why-the-bigger-model-serves-more-context)
- [AWQ quantization](#awq-quantization--the-method-and-why-both-models-use-it)
- [Deployment A — 27B Dense](#deployment-a--qwen36-27b-awq-dense)
- [Deployment B — 35B MoE](#deployment-b--qwen36-35b-a3b-awq-moe)
- [Idle power-down — the LiteLLM sole proxy](#idle-power-down--the-litellm-sole-proxy)

**Part 3 — Measurements & Operations**
- [Prefix caching](#prefix-caching)
- [Experiments that did NOT work](#experiments-that-did-not-work-do-not-retry-without-reason)
- [Hardware ceiling facts](#hardware-ceiling-facts)
- [Auto-compaction on the Qwen proxy](#auto-compaction-on-the-qwen-proxy)
- [Proxy-side compaction (repeated, past the per-session breaker)](#proxy-side-compaction-repeated-past-the-per-session-breaker)
- [Gotchas](#gotchas)

**Part 4 — Scaling Beyond One A10**
- [PCIe bandwidth](#pcie-bandwidth--grounding-the-tp2-on-x4-question)
- [Two A10 cards — TP=2 projections](#two-a10-cards--tp2-projections-for-dense-and-moe)

## Part 1 — Getting Started

### Prerequisites

| need | why | install |
|---|---|---|
| Docker Engine | runs the vLLM + [LiteLLM](https://github.com/BerriAI/litellm) + [Open WebUI](https://github.com/open-webui/open-webui) stack | [docs.docker.com/engine/install](https://docs.docker.com/engine/install/) |
| NVIDIA Container Toolkit | passes the A10 through to the vLLM container | [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| [uv](https://docs.astral.sh/uv/) | bootstraps an isolated `.venv` for the pytest suite — `make test` runs `make ci` (`uv venv` + install `requirements-test.txt`) on first use; nothing is installed in `$HOME` | [install uv](https://docs.astral.sh/uv/getting-started/installation/) |
| python3 | the coding-session benches (`make bench`) | system python ≥ 3.10 |

### Quick Start

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

`run`/`start`/`stop` target the 27B dense stack (`docker-compose.yaml`);
`run35`/`start35`/`stop35` target the 35B MoE stack (`docker-compose.moe.yaml`).
The two stacks share host port `4000` (the LiteLLM proxy) and the one GPU, so stop
one before starting the other.

- LiteLLM proxy (host-facing): `http://localhost:4000` (or the box's LAN IP)
  - Anthropic `/v1/messages` **and** OpenAI `/v1/*` — same base
  - three model names per stack: `<base>` (default), `<base>-preserve`, `<base>-nothink`
    — `<base>` is `qwen3.6-27b` (dense) / `qwen3.6-35b-a3b` (MoE)
- Claude Code: `./bin/claude-qwen [--model moe|dense]` (see *Drive it with Claude Code*)
- opencode: `./bin/opencode-qwen [--model moe|dense]` (see *Drive it with opencode*)
- open-webui: `http://localhost:3000/`
- vLLM direct (metrics + harness): `http://localhost:8000` — LAN-open; `/metrics` (prefix-cache hit%) + raw OpenAI API for test harnesses. Chat clients use the proxy (`:4000`), not this.

The coding-session bench (`scripts/coding_session_bench.py`) streams a chat that
grows one ~2k-token user message + ~500-token assistant reply per turn and
reports per-turn prompt/seq/cached/uncached tokens, TTFT, prefill TPS, output
TPS, and prefix-cache hit %. Cache hit is read from vLLM `/metrics` (at the
**root**, not `/v1`).

### Repo structure

| file | what |
|---|---|
| `docker-compose.yaml` | vLLM (27B Dense) + LiteLLM sole proxy + open-webui stack |
| `docker-compose.moe.yaml` | vLLM (35B MoE, 128k, 2.2 GiB offload) + LiteLLM stack |
| `litellm_config.yaml` / `.moe.yaml` | LiteLLM config per stack: three model names, one shared backend (env-driven) |
| `litellm_callbacks.py` | custom callback: wake-on-request + idle-stop |
| `bin/claude-qwen` | Claude Code wrapper: `--model {moe|dense}` (default moe), sets `ANTHROPIC_*` env, execs `claude` against the proxy |
| `bin/opencode-qwen` | opencode wrapper: `--model {moe|dense}` (default moe), embeds the litellm provider config via `OPENCODE_CONFIG_CONTENT` and execs `opencode -m litellm/<line>` against the proxy |
| `Makefile` | `run` / `start` / `stop` / `run35` / `start35` / `stop35` / `ci` / `test` / `test-integration` / `test-pcie` / `bench` / `bench35` / `bench_pcie` / `idle-test` / `litellm-logs` / `litellm-logs35` |
| `scripts/coding_session_bench.py` | growing coding-session bench against vLLM (prefill/output tps; prefix-cache hit via `/metrics`). Defaults to vLLM (`:8000`) for real hit%; via the proxy (`:4000`) `hit%` reads 0 |
| `scripts/pcie_bw_bench.py` | GPU↔host PCIe D2H/H2D bandwidth + TP=2 all-reduce estimate (in-container via `make bench_pcie` / `make test-pcie`) |

### Security

The proxy binds `0.0.0.0:4000` (LAN-reachable) with auth **disabled**. It does
**not** mount the Docker socket directly — a `docker-sock-proxy` sidecar holds
`/var/run/docker.sock` and exposes only `POST /containers/<name>/{start,stop}`
(+`GET /_ping`) over an internal network; every other Engine call
(create/delete/exec/images/secrets/volumes/networks) returns 403. vLLM is also published on
`0.0.0.0:8000` (raw OpenAI API + `/metrics`; unauthenticated inference). Treat any
host that can reach `:4000` or `:8000` as trusted — this is deliberate; see
[Tuning & security](#tuning--security) for the full exposure and how to lock it down.

## Part 2 — Single-A10 Deployments

### TL;DR — one A10, both models

| | Qwen3.6-27B-AWQ (**Dense**) | Qwen3.6-35B-A3B-AWQ (**MoE**) |
|---|---|---|
| architecture | hybrid: 48 GDN linear + 16 full-attn / 64 layers | hybrid: 30 GDN linear + 10 full-attn / 40 layers |
| total params | 27B | 35B |
| **active params / token** | **27B (all)** | **3B (1 shared + routed experts)** |
| hidden dim | 5120 | 2048 |
| full-attn layers | 16 | 10 (1-in-4) |
| attn / KV heads / head_dim | 24 / 4 / 256 | 16 / 2 / 256 |
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

### Dense vs MoE — why the bigger model serves more context

**The 35B model serves 128k context while the 27B only serves 64k — despite
the 35B having more weights.** Three facts invert the "bigger model = less
context" intuition:

| lever | Dense 27B | MoE 35B-A3B | effect on context ceiling |
|---|---|---|---|
| active params / token | 27B (every MLP runs) | 3B (only routed experts run) | MoE: lighter compute, but **irrelevant to context** (context is KV-bound, not compute-bound) |
| full-attn layers | 16 | 10 (1-in-4) | MoE has fewer layers attending over the prefix |
| hidden dim | 5120 | 2048 | MoE KV head is 2.5× smaller per layer |
| **KV + state per token** | **~35 KiB** (large; ~3× MoE) | **~11 KiB** (but far fewer full-attn tokens) | MoE pays less context cost per token |
| weights on GPU | 18.83 GiB (fits) | ~21.5 GiB (does **not** fit) | MoE must offload 2.2 GiB |

The dense 27B is **param-light enough to fit cleanly** (18.83 GiB weights +
~2.15 GiB KV+state ≈ 21.0 GiB, under the 21.40 GiB 0.97-budget → 64k) but its
16 full-attn layers × hidden 5120 make each
context token expensive, so 64k is the no-offload ceiling.

The MoE 35B is **param-heavy on weights** (21.5 GiB > free GPU) but its 10
full-attn layers × hidden 2048 make each context token cheap. Once 2.2 GiB of
the weights is offloaded to host RAM (UVA — zero-copy GPU-direct PCIe read),
the freed GPU holds ~150k tokens of KV+state → 128k context with margin.

**MoE has no free lunch:** the 2.2 GiB offload is a serving-time tax — a
100 %-CPU UVA offloader thread and the GPU PCIe-stalled to ~80 % power
(120/150 W). See [Offload efficiency cost](#offload-efficiency-cost). The dense
27B pays none of this.

#### MoE no-offload math (why offload is required, not optional)

Budget at util 0.97 = 0.97 × 22.06 = 21.40 GiB. Post-repack weights ≈ 21.38 GiB.
From the 1 GiB offload run: GPU weights 20.31, KV 0.65 → vLLM's non-KV reserve
= (21.40 − 20.31) − 0.65 ≈ 0.44 GiB (mamba state + activation scratch + safety).

| path | weights | reserve | total | budget headroom (per-row util) | usable context |
|---|---|---|---|---|---|
| GPU-only @ 0.97 | 21.38 | 0.44 | 21.82 | over by 0.42 (vs 21.40; Marlin repack OOM) | ~0 |
| GPU-only @ 0.99 | 21.38 | 0.44 | 21.82 | under by ~0.02 (vs 21.84; fails init) | ~2k |
| **2.2 GiB offload @ 0.95** | 19.18 | — | 19.18 | 1.78 GiB KV+state (measured, 150 349 tok; 20.96 − 19.18) | **128k** |

So GPU-only is param+reserve-bound by ~0.2 GiB at the achievable util; offload
is required for any usable context. (The offload is a *serving-time* offload —
it persists into serving, not load-only — but the penalty is ~15 tps, not 2 tps.)

### Hardware

| | |
|---|---|
| GPU | NVIDIA A10, Ampere sm86, 24GB VRAM (~22.06 GiB usable by vLLM) |
| PCIe | Gen4 **x4** — eGPU rig: A10 in an external enclosure, host↔GPU routed via an M.2 NVMe → OCuLink (SFF-8612) adapter, which exposes only 4 lanes. Card is x16-capable; the adapter/wiring caps the link at x4. |
| TP | 1 (single GPU) |
| 27B weights on GPU | 18.83 GiB |
| 27B KV + state cache | ~2.15 GiB → 64 744 tokens |
| 35B MoE weights on GPU | ~19.2 GiB (after 2.2 GiB offload from ~21.4 GiB post-repack) |
| 35B MoE KV + state cache | ~1.78 GiB → 150 349 tokens |

A second A10 on the same host would lift both ceilings and remove the MoE
offload tax — see [Two A10 cards — TP=2 projections](#two-a10-cards--tp2-projections-for-dense-and-moe).

### AWQ quantization — the method, and why both models use it

Both Qwen3.6 checkpoints served here are **AWQ-INT4** (verified in each
`config.json`: `quant_method: awq`, `bits: 4`, `group_size: 128`, asymmetric
zero-points) running on vLLM's **Marlin** kernel. This is the single biggest
reason a 27B and a 35B model both fit on 24GB at near-FP16 quality — and it is
strictly better than naive (static) INT4.

#### The loss problem: INT4 is coarse, and round-to-nearest wastes the budget

INT4 gives each weight one of **16 levels**. Rounding a continuous FP16 weight
into 16 buckets is inherently lossy, so the question is *where* you accept the
error. **Static / round-to-nearest (RTN)** quantization treats every channel the
same — but a model's accuracy is carried disproportionately by a small (~0.1 %)
set of **salient** weight channels (outliers). RTN rounds those high-value
channels just as coarsely as the rest, so the error lands exactly where it hurts
most. At INT4, RTN alone collapses accuracy; staying usable usually means keeping
so many layers in FP16 that you lose most of the memory win.

#### AWQ's idea: protect the salient channels with a cheap per-channel scale

AWQ (Activation-aware Weight Quantization, Lin et al. 2023) finds the salient
channels from the **activations** that run through them — measured on a small
calibration set — rather than from the weights alone. It protects them not by
keeping them in FP16 (expensive) but by applying a **per-channel scale `s`**:
multiply the salient weight channels by `s` (so their fine values survive INT4
rounding) and divide the matching activations by `s` (the math is identical). The
rounding error is pushed onto the unimportant channels, where it is harmless.
The whole tensor stays INT4; only a small per-group scale + zero-point is added
(here, group size 128).

#### How the checkpoint is produced

1. **Calibrate** — run a few hundred representative prompts through the FP16
   model; record per-channel activation magnitudes.
2. **Scale search** — for each linear layer, grid-search the per-channel scale
   `s` that minimizes output reconstruction error (activation-magnitude-weighted).
3. **Quantize** — apply the scales (`W·s`, quantize; activations get `÷s`), pack
   to INT4 with group-128 scales + zero-points (asymmetric).
4. **Protect** — keep a small allowlist in FP16 (`modules_to_not_convert`). For
   the 27B this is the ~5 GiB FP16-leftovers gap shown in the
   [Deployment A footprint table](#deployment-a--qwen36-27b-awq-dense).

Net: AWQ-INT4 typically stays within **<1 perplexity point** and **<1 %** on
standard benchmarks of the FP16 original — at 4 bits/weight. RTN at the same
width is far worse.

#### The repack: AWQ checkpoint → Marlin (W4A16) for vLLM

The AWQ checkpoint on disk is in AWQ's own layout (group scales/zero-points
interleaved with the packed weights). vLLM's fast serving path is the **Marlin
W4A16** kernel — a fused dequant + INT4-GEMM — which needs the weights in a
different, reordered tiling. So at load vLLM **repacks** AWQ → Marlin: the weight
tensor is reordered, scales/zero-points reorganized into one packed tensor the
kernel reads directly, and dequant happens *inside* the GEMM (no separate dequant
pass). The dense repack is automatic; the **MoE** repack
(`awq_marlin_moe_repack`) needs ~128 MiB scratch with raw + Marlin coexisting —
which is exactly why the MoE run **offloads 2.2 GiB before the repack** (see
[MoE no-offload math](#moe-no-offload-math-why-offload-is-required-not-optional)).

The payoff: weights stay INT4-packed in GDDR6 through all of serving. For the
27B that is ~18.8 GiB instead of ~54 GiB FP16 (~2.9× smaller), read at INT4
bandwidth (~3–4× faster decode). Without AWQ-Marlin neither model fits; with it,
both decode flat to max context.

#### Both models, one recipe

| | Dense 27B | MoE 35B-A3B |
|---|---|---|
| AWQ config | 4-bit, group-128, zero-point | 4-bit, group-128, zero-point |
| quantized to INT4 | every layer's MLP `gate`/`up`/`down` | the 256 routed experts |
| kept FP16 (`modules_to_not_convert`) | `self_attn` & `linear_attn` projections, `model.layers.0`, `mtp`, `visual` | whole `self_attn` & `linear_attn`, **+ `shared_expert` + router (`mlp.gate`)**, `model.layers.0`, `mtp`, `visual` |
| repack | automatic | `awq_marlin_moe_repack` — ~128 MiB scratch → forces the 2.2 GiB offload |

Same AWQ-INT4 + Marlin recipe; the only model-specific wrinkle is the MoE
repack's transient scratch, which is what makes offload mandatory for the 35B
(the 27B needs none). Both decode flat to their max context — dense ~21 tps at
64k, MoE ~15.4 tps at 128k — precisely because AWQ-Marlin keeps their weights
small enough to leave room for KV.

### Deployment A — Qwen3.6-27B-AWQ (Dense)

#### Why it fits on 24GB: AWQ + hybrid linear attention + tuned caches

Three things together make 64k context on a 24GB GPU possible — and keep decode
fast at long context.

##### 1. AWQ INT4 weights on the fast Marlin kernel

`QuantTrio/Qwen3.6-27B-AWQ` is genuinely packed INT4 (not just a name), and vLLM
runs it on **AWQ-Marlin** (fused dequant + INT4 GEMM) — *not* dequantized to FP16.
For the quantization method itself (calibration, per-channel scales, the Marlin
repack) see [AWQ quantization — the method, and why both models use it](#awq-quantization--the-method-and-why-both-models-use-it); this section covers the on-GPU footprint.

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
decode would be ~3–4× slower. Marlin keeps INT4 packed in GDDR6 and dequants inside
the GEMM, so you get the full memory/bandwidth benefit.

##### 2. Hybrid architecture — 48 of 64 layers are linear (GatedDeltaNet)

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

##### 3. KV + state caches tuned for the hybrid mix

| cache | dtype | covers | why |
|---|---|---|---|
| KV cache | **fp8** | 16 full-attn layers | FlashInfer on Ampere; halves KV in GDDR6 |
| mamba state cache | **float16** | 48 linear-attn layers | **key lever** — config defaults FP32; FP16 lifts the no-offload ceiling 54.9k → 72.5k tokens |
| prefix cache | on (align mode) | cross-request reuse | experimental for hybrid; 800-token block size |

Without the `--mamba-ssm-cache-dtype float16` lever, 64k context would not fit
on this GPU. See [Working config (27B)](#working-config-27b).

#### Working config (27B)

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
- `--tool-call-parser qwen3_xml`
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
| `--tool-call-parser` | qwen3_xml | Qwen3.6 native XML tool calls |
| `--trust-remote-code` | — | hybrid arch needs it |

Env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`. HF cache mounted read-only; `HF_TOKEN` resolves from
host env (not committed).

#### Verified performance (27B)

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

### Deployment B — Qwen3.6-35B-A3B-AWQ (MoE)

#### Architecture

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
**fewer layers (40 vs 64), fewer full-attn layers (10 vs 16), and a 2.5× smaller
hidden (2048 vs 5120)** — that is the source of its larger context ceiling.

#### Why offload is required (measured envelope)

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

**Why nightly.** Both composes pin the same image, `vllm/vllm-openai:nightly@sha256:9fe761ad…`.
The MoE requires nightly for `awq_marlin_moe_repack` (the 35B-A3B Marlin repack above); the
dense 27B — previously `v0.24.0` — rides the same image so both stacks run one vLLM version
(re-validated on nightly: `make bench` hit% 0→78, out_tps ≈ 21).

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

#### Working config (35B MoE)

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
- `--tool-call-parser qwen3_xml`
- `--reasoning-parser qwen3`
- `--trust-remote-code`

Env/volumes match the 27B compose (HF cache ro, `vllm_cache_moe` volume,
`HF_TOKEN` from host env, `expandable_segments:True`).

#### Verified performance (35B MoE)

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

#### Offload efficiency cost

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

#### MoE checkpoint status

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

### Idle power-down — the LiteLLM sole proxy

Both stacks put **LiteLLM** (`ghcr.io/berriai/litellm:main-stable`) in front of
vLLM as the single host-facing service on `0.0.0.0:4000`. It owns four things:

1. **Translation** — Anthropic `/v1/messages` ↔ OpenAI `/v1/chat/completions`, so
   Claude Code drives local Qwen3.6 as if it were Anthropic and opencode drives
   it as OpenAI (see *Drive it with Claude Code* / *Drive it with opencode*).
   LiteLLM also normalizes the `ctx`/`msg`/`system` roles newer
   Claude Code injects, which vLLM's native `/v1/messages` rejects with 400.
2. **Three model names** — against one vLLM instance, differing only in
   `chat_template_kwargs` (see *Thinking*); switchable via `/model`.
3. **The vLLM lifecycle** — a custom callback (`litellm_callbacks.py`) wakes
   the backend on request and `docker stop`s it after 15 min of inference-idle.
4. **Web search** — LiteLLM's `websearch_interception` converts Claude Code's
   native `web_search` tool to a SearXNG (DuckDuckGo) lookup and feeds the
   result back for Qwen to synthesize, so WebSearch works under Qwen with no
   client setup (see *Web search*).

**Goal, met: after 15 min idle the A10 draws ~15 W (< 50 W target).**

**Why stop/start, not vLLM sleep mode.** vLLM's sleep mode
(`--enable-sleep-mode`, `POST /sleep?level=1`) frees VRAM but keeps the process
and CUDA context resident — that is exactly what makes wake-up fast, but a live
context holds the GPU out of its lowest P-state. An *asleep* vLLM still draws
~45–60 W (borderline vs < 50 W, and version-fragile — sleep-mode memory-freeing
has regressed before). Only stopping the container reaches the reliable floor,
so stop/start is the chosen mechanism. Sleep mode was tested for the record; it
is **not** wired into either stack.

**Measured (A10, this session):**

| condition | power | memory | p-state |
|---|---|---|---|
| served / loaded-idle (both stacks) | ~58 W | ~22 GiB | P0 |
| idle 15 min → backend stopped | **~15 W** | **0 MiB** | P8 |

The `0 MiB` reading proves the CUDA context is torn down, not merely asleep.
After a stop the card re-downshifts P0→P8 within ~30 s, then settles to ~15 W.

| latency | 27B Dense | 35B MoE |
|---|---|---|
| first-ever cold load | 129 s | 168 s |
| wake after idle (OS caches warm) | 32 s | 53 s |
| `WAKE_TIMEOUT_SECONDS` budget | 240 s | 300 s |

**Topology.** The `vllm` service publishes `0.0.0.0:8000:8000` (LAN-open — see
*Security*): `/metrics` (prefix-cache hit%) and the raw OpenAI API for test
harnesses/overhead measurement. Chat clients do **not** use it — they go through
the proxy. LiteLLM still reaches vLLM over the compose network at
`http://vllm:8000` and binds `0.0.0.0:4000:4000` (LAN-accessible). open-webui points at
`http://litellm:4000/v1`. `make bench` hits vLLM directly
(`http://localhost:8000`) so its `hit%` is real; `bin/claude-qwen` /
`bin/opencode-qwen` hit the proxy (`http://localhost:4000`). Remote hosts hit the box's LAN IP (override
`ANTHROPIC_BASE_URL` / `OPENAI_API_BASE_URL`).

**Service topology** (dense; the MoE stack is the same shape, different model and
container names). `litellm` is the hub — the only service on both networks:

```
                      LAN
                       |   :4000 litellm  ·  :3000 open-webui  ·  :8000 vllm (metrics/bench)
                       v
                  +----------+
                  | litellm  | :4000  -- the hub (only service on BOTH networks):
                  | (hub)    |        translate + wake/idle lifecycle + websearch_interception
                  +----+-----+
       infer /v1/chat  |        | web_search tool -> SearXNG
        +-------------+        +--------------+
        v                                     v
   +----------+                         +------------+   HTTPS   +------------+
   |  vllm    |                         |  searxng   |  ------>  | DuckDuckGo |
   | :8000 GPU|                         |   :8080    |           | (internet) |
   +----------+                         +------------+           +------------+

   open-webui :3000  (browser chat UI)  -->  litellm:4000/v1

   docker-api network (INTERNAL, no egress):
       litellm  --Engine API-->  docker-sock-proxy :2375  -->  /var/run/docker.sock
                                (HAProxy whitelist: POST /containers/{id}/{start,stop} + GET /_ping only)

   default network (bridge, HAS egress): litellm + vllm + open-webui + searxng.
```

#### Endpoints

Base URL `http://<host>:4000` — `<host>` is `localhost` on the box, its LAN IP
from a remote host. Auth is **disabled** on all of these (no `master_key`); the LAN
is the trust boundary (see *Security*).

| Method & path | Wakes GPU? | Use |
|---|---|---|
| `POST /v1/messages` | **yes** | Anthropic Messages API — `bin/claude-qwen` / Claude Code |
| `POST /v1/chat/completions` | **yes** | OpenAI Chat — open-webui, `make bench` |
| `POST /v1/completions` | **yes** | OpenAI legacy completions |
| `GET /v1/models` | no (cold) | lists the three model names; safe for pollers (open-webui, gateway discovery) |
| `GET /health/liveliness` | no | process-alive probe (`200 "I'm alive!"`, `503` while draining) |
| `GET /health/readiness` | no | config + DB validity; safe for load balancers |
| `GET /metrics` | no | LiteLLM's own Prometheus metrics (proxy-side; not vLLM's `/metrics`) |

The completion routes wake the backend before the first byte — the router path
(`/v1/chat/completions`, `/v1/completions`) via `async_pre_call_hook`, the
Anthropic pass-through (`/v1/messages`) via `async_pre_request_hook`. The cold
routes are served from config / process state and never touch the GPU — that is
why open-webui's model-list polling and Claude Code's gateway discovery cannot
pin the card awake. The deep `GET /health` (not in the table) pings the backend
directly, so it reports **unhealthy while vLLM is idle-stopped**; it carries no
chat traffic and will not revive a stopped backend on its own — use
`/health/liveliness` for an alive probe.

**Wake is inference-triggered.** Both pre-call hooks run `ensure_up()` before
every completion (coalesced behind a lock, so a burst wakes once) and fire only
on real LLM calls — `/v1/models` and `/health` are served cold from config, so
background pollers cannot pin the GPU awake. While the backend is down,
`/v1/models` still lists all three model names; wake fires on the first message.

**Idle detection.** A background task (started via
`LITELLM_WORKER_STARTUP_HOOKS=litellm_callbacks:start_background_tasks`) polls
`http://vllm:8000/metrics` (at the **root**, not `/v1`) every `POLL_SECONDS` and
parses `vllm:num_requests_running`, `_waiting`, `_swapped`. Idle = all three
`== 0` sustained `IDLE_SECONDS` (one busy poll resets the timer).
`num_requests_swapped` is absent in vLLM v1 and is treated as 0. On idle →
`POST /containers/{name}/stop`. At startup a `bootstrap()` task waits for the
warm-started backend's `/health` and arms the timer — without it, a
freshly-started-but-unused stack would never idle-stop. Container control talks to
the Engine API via the `docker-sock-proxy` sidecar (`DOCKER_API_BASE`) over `httpx`
(the litellm image ships no docker SDK); the sidecar whitelists only `start`/`stop` + `/_ping`.

#### Thinking — three model names, one backend

`preserve_thinking` and `enable_thinking` are **per-request** Qwen3.6 Jinja flags
(verified in `QuantTrio/Qwen3.6-*/chat_template.jinja` — the 27B and MoE
templates are byte-identical). One running vLLM serves all combos; LiteLLM sends
the right `chat_template_kwargs` per model name (`<base>` = `qwen3.6-27b` dense
or `qwen3.6-35b-a3b` MoE):

| model name | `preserve_thinking` | `enable_thinking` | use |
|---|---|---|---|
| `<base>` | false | true (default) | **DEFAULT** — thinking shown this turn, not carried |
| `<base>-preserve` | true | true (default) | **RARE** — carry prior `<think>` across turns |
| `<base>-nothink` | false | false | **BACKGROUND** — no reasoning pass (titles/summaries) |

The template always keeps the **most-recent** turn's reasoning; `preserve_thinking`
additionally keeps *older* turns. With `--reasoning-parser qwen3`, vLLM emits
`reasoning_content`, which LiteLLM's `/v1/messages` adapter renders as native
Anthropic `{type:thinking}` blocks (the collapsible thinking UI).

**`-preserve` works end-to-end.** The Qwen3.6 chat template renders prior-turn
reasoning only from `reasoning_content`, but on the `/v1/messages` path LiteLLM's
Anthropic→OpenAI adapter attaches `thinking_blocks` instead (and rebuilds prior
assistant turns *after* the hook fires). So an `async_pre_call_deployment_hook`
normalizes `thinking_blocks`/`reasoning` → `reasoning_content` on prior assistant
turns before vLLM renders them. Verified on `/v1/messages` (the Claude Code path):
a 2-turn `-preserve` request carries ~44 extra prompt tokens of prior reasoning
that the default strip variant drops.

#### Web search

Under the Qwen stack WebSearch would be inert — it is an Anthropic server-side
tool that does nothing against a custom base URL — so LiteLLM's built-in
`websearch_interception` callback runs it **server-side**: it converts Claude
Code's native `web_search` tool to `litellm_web_search`, runs the lookup through
a self-hosted **SearXNG** instance (DuckDuckGo engine, **no API key**), feeds the
results back, and Qwen synthesizes a grounded answer with citations. The client
needs **no change** — `bin/claude-qwen` is plain `claude` with the Qwen env (no
MCP server, no `--disallowedTools`).

Real Claude Code requests carry `web_search` *alongside* their other tools (Read,
Bash, …), so they take LiteLLM's **agentic loop**: Qwen calls the search tool,
SearXNG runs, results return, Qwen synthesizes. That is why this works despite
upstream [litellm#29649](https://github.com/BerRIAI/litellm/issues/29649), whose
raw-results short-circuit fires only for web-search-*only* requests — a shape
Claude Code never sends. Verified end-to-end: a time-sensitive question returns a
synthesized answer citing a source URL.

Config (`litellm_config.yaml` / `.moe.yaml`, both stacks): `callbacks:
[websearch_interception]` gated to the Qwen proxy by
`websearch_interception_params.enabled_providers: [hosted_vllm]` (the owner's
real `claude` → Anthropic never hits this proxy), plus a `search_tools` SearXNG
entry reading `SEARXNG_API_BASE`. The `searxng` compose service sits on the
**default** network (it needs internet egress to reach DuckDuckGo — *not* the
internal no-egress `docker-api` net); `searxng/settings.yml` enables the JSON
output format LiteLLM queries and disables the bot limiter. Needs LiteLLM
≥ v1.78.7 (`main-stable` is well past it).

#### Drive it with Claude Code

```bash
make start35 && ./bin/claude-qwen                  # 35B MoE stack (default)
make start   && ./bin/claude-qwen --model dense    # 27B dense stack
```

`--model {moe|dense}` (default `moe`) picks the Qwen line and must match the
running stack — start it first (`make start35` MoE / `make start` dense); the two
stacks share one A10 + host `:4000`, so only one runs at a time and switching
lines is a manual cold swap, not automatic. `claude-qwen` points Claude Code at the proxy
(`ANTHROPIC_BASE_URL=http://localhost:4000`, no `/v1`), sends a placeholder
`ANTHROPIC_AUTH_TOKEN` (the proxy is auth-free; Claude Code just needs one
non-empty), sets the model to `qwen3.6-27b` / `qwen3.6-35b-a3b` and the
background model to its `-nothink` variant, and enables gateway model discovery
so `/model` lists all three flavors. It also tells Claude Code the *real* upstream
window (`CLAUDE_CODE_AUTO_COMPACT_WINDOW` = 128k MoE / 64k dense) and caps output
(`CLAUDE_CODE_MAX_OUTPUT_TOKENS` = 16384 MoE / 8192 dense): behind a gateway, Claude
Code otherwise assumes a 200k window and sends `max_tokens=32000`, so auto-compaction
is scheduled past the 128k/64k wall and the request overflows (a 400 `ContextWindowExceeded`).
The proxy's `CLAUDE_QWEN_MAX_TOKENS_CAP` (above) is the server-side backstop for
subagents/the small-fast model, which ignore the client env. Sizing the window is
necessary but not sufficient — the proxy must also report accurate streamed usage for
compaction to actually fire; see [Auto-compaction on the Qwen proxy](#auto-compaction-on-the-qwen-proxy). Switch mid-session
`/model qwen3.6-27b-preserve`, or set `QWEN_FLAVOR=preserve` before launch. Drive
a remote box with `CLAUDE_QWEN_BASE_URL=http://<ip-or-hostname>:4000 ./bin/claude-qwen`
(`<ip-or-hostname>` is the box's LAN IP or hostname).
The host needs the `claude` CLI on `PATH`.

#### Drive it with opencode

```bash
make start35 && ./bin/opencode-qwen                  # 35B MoE stack (default)
make start   && ./bin/opencode-qwen --model dense    # 27B dense stack
```

`--model {moe|dense}` (default `moe`) picks the Qwen line and must match the
running stack — start it first (`make start35` MoE / `make start` dense); the two
stacks share one A10 + host `:4000`, so only one runs at a time and switching
lines is a manual cold swap, not automatic. opencode ignores `OPENAI_BASE_URL`
for its built-in `openai` provider, so the wrapper hands it a dedicated
openai-compatible provider by embedding the `litellm` provider config (baseURL,
apiKey, the line's three model variants with context/output limits) as a static
JSON literal passed via `OPENCODE_CONFIG_CONTENT`. opencode deep-merges that as
local scope on top of your global `~/.config/opencode/opencode.json` — global
`mcp` servers (codebase-index/deepwiki/agentmail) are preserved, and only
`provider`/`model`/`small_model` are overridden to the local Qwen line. No
sidecar file, no cache dir, no repo-path lookup — the wrapper is self-contained
and works unchanged from the `~/bin` copy. Pick the line with `--model
{moe|dense}` and the flavor with `QWEN_FLAVOR` (default|preserve|nothink), or
switch mid-session via `/model litellm/qwen3.6-<line>[-preserve|-nothink]`; the
`-nothink` variant is the config's `small_model`. Context/output caps match the
proxy (`limit.context` 64k dense / 128k MoE, `limit.output` 8192 dense / 16384 MoE). Drive a
remote box with `OPENCODE_QWEN_BASE_URL=http://<ip-or-hostname>:4000 ./bin/opencode-qwen`
(`<ip-or-hostname>` is the box's LAN IP or hostname; the wrapper exports the localhost
default itself; opencode has no
`{env:VAR:-default}` form).

#### Tuning & security

| env (on the `litellm` service) | default | meaning |
|---|---|---|
| `VLLM_IDLE_SECONDS` | 900 | sustained idle before stop |
| `VLLM_POLL_SECONDS` | 10 | `/metrics` poll interval |
| `VLLM_WAKE_TIMEOUT` | 240 (27B) / 300 (MoE) | cold-start health-wait budget |
| `CLAUDE_QWEN_MAX_TOKENS_CAP` | 16384 | server-side `max_tokens` clamp (8192 dense / 16384 MoE). Claude Code's gateway discovery ignores token limits, so it sends its built-in `max_tokens=32000` on every path — incl. subagents and the small/fast model, which ignore `CLAUDE_CODE_MAX_OUTPUT_TOKENS`. The deployment hook in `litellm_callbacks.py` caps it before vLLM on both endpoints; this env sets the per-stack ceiling. Must satisfy `PCT×WINDOW + cap ≤ WINDOW` (PCT=80 → cap ≤ WINDOW/5): dense 8192 ≤ 12800, MoE 16384 ≤ 25600. |

`make litellm-logs` / `make litellm-logs35` tail the proxy (wake/stop events log
here). The idle-stop is a deliberate `docker stop` (Engine API), so the backend
stays exited until the next request — `restart: unless-stopped` honors an
API-initiated stop (only a crash auto-restarts). LiteLLM itself is also
`unless-stopped`.

**Security — deliberate exposure.** LiteLLM is bound `0.0.0.0:4000` (reachable
from the LAN, by design) and auth is **disabled** (no `master_key`: open-webui,
the bench, and remote Claude Code hosts need no API key). The service does **not**
mount the Docker socket. Instead a `docker-sock-proxy` sidecar — on a dedicated
internal network that only LiteLLM joins — holds `/var/run/docker.sock` and
whitelists `POST /containers/<name>/{start,stop}` plus `GET /_ping`; every other
Engine call (create/delete/exec/images/secrets/volumes/networks) returns 403. Net
effect: a caller who compromises LiteLLM (or any host that reaches `:4000` and
achieves code exec there) can start/stop **any existing container** on this box
(host-wide DoS; privilege recovery if a privileged/host-mounted container exists)
but can no longer create containers, exec into them, pull images, or read secrets —
the privileged-container root-escape (`POST /containers/create` with a host bind)
is closed. vLLM's `0.0.0.0:8000` is the same LAN posture but lighter: it exposes
the OpenAI API and `/metrics` (unauthenticated inference + prefix-cache counters),
not the Docker socket. To put a key in
front later, set `master_key` in `litellm_config.yaml` and point clients'
`ANTHROPIC_AUTH_TOKEN` / `OPENAI_API_KEY` at it; that gates the API. Tighten to
`127.0.0.1:4000` once you stop needing remote access.

**All state is in-repo** — no systemd, cron, or `nvidia-smi -pm 1`. The idle
layer is the `litellm` compose service plus `litellm_callbacks.py`; both the
config and the callback are mounted read-only. The proxy is additive — remove
the `litellm` service from the composes (and the wrappers/JSONs/callback) to
return to a direct-vLLM topology.

---

## Part 3 — Measurements & Operations

### Prefix caching

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
- **Measure via `/metrics` at the root**, not `/v1/metrics`. vLLM publishes
  `:8000`, so read it directly: `curl -s localhost:8000/metrics` (or
  `docker compose exec vllm curl -s localhost:8000/metrics` from inside the net). Counters:
  `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total` (cumulative
  token counts). `usage.prompt_tokens_details` is `null`, so usage
  can't be used for hit rate. `make bench` defaults to vLLM (`:8000`), so its
  `hit%` column is real; point it at the proxy (`:4000`) and `hit%` reads 0
  (LiteLLM's `/metrics` has no vLLM counters — TPS/TTFT still valid).

#### The reported `hit_rate` lags the real per-turn hit

vLLM v1's `CachingMetrics.hit_rate` (stats.py:107) is
`aggregated_query_hit / aggregated_query_total` over a **rolling deque of the
past N requests** — it is a lagging moving average, not the current request's
hit rate. During the 35B bench the *reported* rate reads ~36 % while the
per-turn hit (from `/metrics` deltas) is already ~90 %+. By the end of the run
the reported rate converges to ~98 %. A low reported `hit_rate` mid-run is not a
bug — the per-turn bench column is authoritative, or wait for the rolling window to fill.

### Experiments that did NOT work (do not retry without reason)

#### MTP speculative decoding — does not fit on this GPU (27B)

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
(~3 GiB = FP16 embedding ~1.56 + CUDA ctx ~0.6 + cuDNN workspaces ~0.5 + activation/scratch ~0.35) ≈ 21.84
GiB, leaving only ~184–224 MiB. The MTP head needs ~340 MiB. The shortfall is
**independent of `num_speculative_tokens`** (1 vs 2 changes only runtime draft
activations, not the head footprint) and **independent of util** (drafter loads
before KV sizing). `--max-num-batched-tokens` 1024→256 freed only ~40 MiB — the
overhead is fixed, not batch-token-driven.

**Conclusion:** MTP is infeasible on this 24GB A10 with this 18.83 GiB model; a
≥40 GB GPU would fit it. Expected gain if it fit: ~1.3–1.7× output TPS
(single-token speculation, 1-layer head, ~50–70 % acceptance) → ~28–35 tps, not 2×.

#### CUDA graphs (removing `--enforce-eager`) — costs more context than it gains

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

#### CPU offload on the 27B (6 GiB) — rejected as too slow

`--cpu-offload-gb 6` gives 128k context easily on the dense 27B but decode
drops to ~2 tps (PCIe weight reads every step — the *whole* 27B reads CPU each
token because every layer is active). Rejected as too slow. This is the opposite
of the MoE path: on the MoE, only the offloaded expert fraction reads CPU and
routing often misses the offloaded experts, so 2.2 GiB costs ~15 tps, not 2.

#### KV-cache offloading (`--kv_offloading_backend`) — wrong lever, not a ceiling-raiser

`--kv_offloading_backend native --kv_offloading_size <GiB>` (vLLM's OffloadingConnector,
RFC #26858) is unrelated to the `--cpu-offload-gb` *weight* offload used on the MoE above.
It ships *completed* prefix-cache blocks to pinned host RAM so a later request sharing that
prefix is restored instead of recomputed — i.e. it extends the **prefix cache**, not the
per-sequence context window. The in-flight sequence's active KV cannot spill to CPU via this
connector, so it raises none of the ceilings in [Hardware ceiling facts](#hardware-ceiling-facts):
not the dense 27B's ~4.0 GiB-for-120k gap, not the MoE's param-bound fit. (The mechanism that
*can* move a running sequence's KV is preemption/swap, not this connector — and at
`--max-num-seqs 1` that stalls decode on the hot path, the same PCIe tax that sank
`--cpu-offload-gb 6` above.) It is also moot at single-user batch 1: no concurrent request
evicts the prefix cache, so the host tier would sit idle. KV offloading is a multi-tenant /
high-throughput feature; neither stack uses it. Do not add it expecting more context.

#### Dead MoE checkpoints

See [MoE checkpoint status](#moe-checkpoint-status) — mattbucci (garbage,
hybrid-quant fusion breaks both runtimes) and cyankiwi (OOM at construction,
param-bound). Do not retry without reason.

### Hardware ceiling facts

- **120k no-offload is impossible on this A10 for the dense 27B**: 22.06 −
  18.83 = 3.23 GiB raw free (0.97 cap → 2.57); at ~35 KiB/token, 120k needs
  ~4.0 GiB — over budget. Real ceiling 64 744 tokens (L152; the bare rate on
  raw free is ~95k, but the 0.97 cap + non-KV overhead lower it to 64k).
- **The MoE 35B is param-bound without offload** — weights 21.38 + reserve 0.44
  = 21.82 GiB > the 21.40 GiB achievable at util 0.97. No amount of context fits
  without `--cpu-offload-gb`; 2.2 GiB offload lifts it to 128k.
- **`--gpu-memory-utilization` above ~0.99 fails** on this A10: free VRAM
  21.83/22.06 = 0.9896, so 0.99×22.06 = 21.84 > 21.83.
- **FP8 KV on Ampere** works via FlashInfer (auto-selected); no native FP8 tensor
  cores needed. `int4_per_token_head` KV is nightly-only; TurboQuant 4-bit KV is
  broken on hybrid Qwen3.5.

### Gotchas

- **open-webui persisted base URL**: `OPENAI_API_BASE_URL` env only seeds a fresh
  DB. The `open-webui-data` volume persists `openai.api_base_urls` in sqlite
  `config`, which takes precedence → empty model dropdown if it still points at
  an old backend. Fix: update that DB row to `http://litellm:4000/v1` and restart,
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

### Auto-compaction on the Qwen proxy

Claude Code proactively auto-compacts a conversation when its running context count
crosses a threshold (~80% of the window set by `CLAUDE_CODE_AUTO_COMPACT_WINDOW`,
per `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`). That running count is grown from the
**streamed `message_start.usage.input_tokens`** of each turn — *not* the terminal
`result.usage`. The LiteLLM→vLLM proxy hardcodes `message_start.usage.input_tokens: 0`
(the real count arrives only in the terminal `message_delta`, because vLLM, like OpenAI,
emits usage in the final streaming chunk), so Claude Code's tracker stays at 0, the
threshold is never crossed, and compaction **never fires (without the usage fix below)** —
context grows unbounded until a 400 `ContextWindowExceeded`. This is why the window/cap fix alone
(`CLAUDE_CODE_AUTO_COMPACT_WINDOW` + `CLAUDE_QWEN_MAX_TOKENS_CAP`) is necessary but not
sufficient: it sizes the window and caps output, but compaction still will not fire until
the proxy reports accurate streamed usage. (The `~/bin/glm` wrapper works because z.ai's
endpoint reports `message_start` usage correctly — compaction fires there at ~70%.)

The fix is server-side, in `litellm_callbacks.py` (`CLAUDE_QWEN_INJECT_STREAMED_USAGE=1`,
default on): preflight-tokenize the prompt with vLLM `POST /tokenize` and inject that
count into the streamed `message_start`. The count is exact because `/tokenize` applies
the same chat template vLLM uses for generation.

The non-obvious part is *where* the preflight runs. Two hook points were considered:

- `async_pre_call_deployment_hook` — fires **before** LiteLLM converts the Anthropic
  request to the OpenAI shape vLLM expects, so its `messages`/`tools` are pre-conversion.
  `/tokenize` of them undercounts tool-bearing requests (the system-message merge and
  tool-format conversion happen after the hook). For real Claude Code (~28k-token fixed
  tools schema) the gap is large enough to fire compaction late, past the dense stack's
  safety margin.
- `log_pre_api_call` (a logging-path hook, read-only) — fires **after**
  `transform_request` builds the exact wire body and **before** the HTTP POST to vLLM.
  Its `kwargs["additional_args"]["complete_input_dict"]` *is* the post-conversion body
  vLLM receives, so `/tokenize` of it == vLLM's `prompt_tokens` exactly — for tools and
  no-tools alike.

There is no request-path/mutation hook that fires post-conversion on this path
(`async_pre_request_hook` is dispatched only on the outer pass-through handler, on the
*original* Anthropic kwargs; `async_log_pre_api_call` is not dispatched at all). Read-only
is sufficient because the actual mutation is done in the streaming iterator hook, which
rewrites the first `message_start` SSE frame. `log_pre_api_call` runs in a LiteLLM
threadpool worker (not the event loop — `asyncio.get_running_loop()` raises there), so the
sync `/tokenize` (urllib, stdlib — thread-safe across workers, unlike a shared
`httpx.Client`) does not block the loop, and it completes before the POST — so the stashed
count is ready before `message_start` is emitted. The stash is keyed by `litellm_call_id`,
the same value at both hooks (the nested inner `acompletion` reuses the outer call id).

```
        Claude Code   (auto-compact tracker reads message_start.usage.input_tokens)
             │   POST /v1/messages  (stream, Anthropic shape, +tools)
             ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ LiteLLM  /v1/messages   (experimental_pass_through handler)           │
 │                                                                       │
 │  outer async_pre_request_hook ── original Anthropic kwargs            │ ← pre-conversion (not used)
 │          │                                                            │
 │          ▼  adapter: Anthropic → OpenAI                               │   system→system msg,
 │          │                                                            │   tools→tools[] (separate)
 │          ▼  inner litellm.acompletion()                               │
 │  async_pre_call_deployment_hook ── OpenAI kwargs                      │ ← pre-conversion (undercounts tools)
 │          │                                                            │
 │          ▼  transform_request  →  wire body (data)                    │
 │  ★ log_pre_api_call ── complete_input_dict = data                     │ ← POST-conversion  (EXACT)
 │          │      PREFLIGHT: POST vLLM /tokenize                        │   stash count by litellm_call_id
 │          ▼  HTTP POST /v1/chat/completions  (stream)                  │
 └─────────────────────────┬─────────────────────────────────────────────┘
                           ▼
              vLLM   (qwen3.6; /tokenize uses the same chat template)
                           │   SSE stream (OpenAI; usage in the FINAL chunk)
                           ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ LiteLLM  serialize → Anthropic SSE                                    │
 │  ★ async_post_call_streaming_iterator_hook                            │ ← INJECT: rewrite the first
 │          │      message_start.usage.input_tokens = stashed count      │   message_start SSE frame
 │          ▼   event: message_start\ndata: {… "usage": {…}}             │
 └─────────────────────────┬─────────────────────────────────────────────┘
                           ▼
        Claude Code   tracker grows → crosses window − max_output − 13K (~98K at WINDOW=128000;
                       PCT override ignored for gateway models) → compact_boundary fires
```

Verified live (MoE stack, claude 2.1.193 via `/v1/messages`):

- 4-tool request: `message_start.input_tokens` 0 → **538**; terminal `message_delta` 538 (exact).
- no-tools request: 0 → **15**; terminal 15 (exact).
- Same `litellm_call_id` at `log_pre_api_call` and the iterator hook → injection fires.
- **Compaction now fires.** A `WINDOW=128000` probe growing context ~12K/turn produced a
  `compact_boundary` event (`compactMetadata.trigger="auto"`, `preTokens=98453`,
  `postTokens=13629` — 87% reduction, 61s) at the `window − max_output − 13K` threshold
  (128000 − 16384 − 13000 = 98616). The `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80` did **not**
  scale it (fired at the unscaled value) — the override is ignored for gateway models; the
  `~70%` glm figure is not reproduced here. The prior "compaction never fires" symptom was
  the zero-streamed-usage bug, not a separate dispatch failure.

  **Known limitation — one compaction per session.** Claude Code gates proactive
  compaction behind a per-session breaker (`hasAttemptedReactiveCompact`); after the first
  `compact_boundary` it does not fire again even on subsequent threshold crossings (the
  session eventually 400s at the 128K wall if it regrows past). This is a claude-code
  internal, not proxy-fixable, and it persists across `--resume`. The fix above gives one
  ~85K-token reset per session (98K→13K) — a large improvement over "never fires." For
  workloads that need **repeated** proxy-side compaction across that boundary, an opt-in
  `compact_20260112` polyfill in LiteLLM bypasses the claude-code breaker server-side —
  see [Proxy-side compaction](#proxy-side-compaction-repeated-past-the-per-session-breaker)
  below.

To disable (revert to the zero-streamed-usage behavior, e.g. to debug compaction):
set `CLAUDE_QWEN_INJECT_STREAMED_USAGE=0` in the litellm compose env and recreate the
litellm container (`make start35` / `make start`, or `docker compose -f
docker-compose.moe.yaml up -d --force-recreate litellm` to keep vLLM warm).

**Overhead.** Two proxy features add per-request latency:
- **Preflight `/tokenize`** (`CLAUDE_QWEN_INJECT_STREAMED_USAGE=1`, on by default): every streamed `/v1/messages` call issues a `POST /tokenize` to vLLM before generation, producing the exact `input_tokens` for `message_start.usage`. One extra vLLM round-trip per streamed message — on the request's critical path, but offloaded to a LiteLLM threadpool worker (event loop stays free) and prefix-cached, so fast but non-zero.
- **Polyfill recount + summarization** (`CLAUDE_QWEN_PROXY_COMPACT=1`, off by default): recounts input tokens on every `/v1/messages` call, and at each threshold crossing runs a full-history summarization pass on vLLM before the main call. One extra full vLLM generation per compaction — the dominant cost.

### Proxy-side compaction (repeated, past the per-session breaker)

The usage fix above makes claude's own auto-compact fire **once** per session. For long
sessions that regrow past the first reset, claude's per-session breaker
(`hasAttemptedReactiveCompact`) then blocks any further proactive compaction and the
session eventually 400s at the `--max-model-len` wall. The opt-in **proxy-side
compaction** is a server-side workaround that fires on **every** threshold crossing, not
just the first — it is not gated by claude's breaker because it runs entirely inside
LiteLLM, transparent to the client (no `compact_boundary` event is ever emitted).

**How it works.** LiteLLM `main-stable` ships an Anthropic-style `context_management`
edit (`compact_20260112`) with an in-gateway polyfill that runs across all providers,
including `hosted_vllm`: count the request's input tokens, and if they exceed the
trigger, call a separate summarizer model with the full history + a summarization prompt,
then inject the summary as a system prefix and strip the old messages — all before the
main call reaches vLLM. The `async_pre_request_hook` in `litellm_callbacks.py` injects
the `context_management` spec on the `/v1/messages` pass-through path when the opt-in env
is on, and **also sets a per-request `drop_params=false`** (see below) to un-gate the
polyfill for that call only. Because the rewrite happens before our `log_pre_api_call`
`/tokenize` preflight, the injected `message_start.usage.input_tokens` reflects the
**post-compact** count, so claude's tracker resets each cycle — the same mechanism as
outcome A, repeated.

**Opt-in env (litellm service, `docker-compose.{,moe.}yaml`):**

| env | default | meaning |
|---|---|---|
| `CLAUDE_QWEN_PROXY_COMPACT` | `0` (off) | `1` injects `context_management` on `/v1/messages` for repeated server-side compaction. |
| `CLAUDE_QWEN_PROXY_COMPACT_THRESHOLD` | `90000` (MoE) / `50000` (dense) | input-token trigger. `≥ 50000` is the polyfill minimum. |

To enable: set `CLAUDE_QWEN_PROXY_COMPACT: "1"` and recreate litellm (`make start35` /
`make start`). Off by default — the usage fix (one compaction per session) stays the
default behavior; this layer is for workloads that need repeats.

**Requirements already wired into the configs (both stacks):**

- `general_settings.context_management_summary_model` must be set, else the polyfill
  **silently no-ops** (`applied_edits[0].error = "summary_model_not_configured"` — not an
  HTTP error, a footgun). Set to `qwen3.6-35b-a3b-nothink` (MoE) / `qwen3.6-27b-nothink`
  (dense): a cheap summarizer already in `model_list`. The summary call is a router
  `acompletion` on a different path, so it does not recurse into the polyfill.
- `litellm_settings.drop_params: true` (the global default) stays **true** — it keeps
  dropping claude-code params vLLM rejects to avoid 400s for **every** request. The
  polyfill is un-gated **per-request** instead: `async_pre_request_hook` sets a top-level
  `drop_params=false` on the opt-in `/v1/messages` call, which overrides the global
  `true` for that one call only. Why per-request, not global `false`: the polyfill gate
  reads `effective_drop_params` and short-circuits to no-op when truthy, so it must see
  `false` to run — but global `false` would remove the 400 shield for all non-opt-in
  traffic. The per-request key must be **top-level** in the hook's returned
  `request_kwargs`: the hook's `litellm_params` sub-dict is popped and discarded
  (`messages/handler.py:261`), so a sub-dict key would never reach the gate; top-level
  survives the named pops, merges via `kwargs.update` → `GenericLiteLLMParams(**kwargs)`
  → `litellm_params.drop_params` (read at `messages/handler.py:503`) and overrides the
  global the router set. Verified live (A/B): global `true` alone strips
  `context_management` to null (polyfill no-op); adding the per-request `false` makes the
  polyfill fire — `applied_edits[0]` is a `compact_20260112` edit with a real
  summarization sub-call. The config-consistency test guards the global `true`; the hook
  unit test guards the per-request `false`.

**Verified.** The multi-turn reduction shape was established under the prior global
`drop_params: false` config: a clean isolation probe (`WINDOW=1000000` so claude's own T4
threshold is unreachable — only the polyfill can fire; `~18K`-token user turns, 12 turns)
produced **two** repeated proxy-side compactions (turn 5: 95196→48556; turn 9: 93160→48635),
each with a vLLM summarization burst, `compact=False` throughout (transparent), no 400, no
timeout, claude's tracker resetting each cycle. The H1 change (global `true` + per-request
`false` in the hook) only swaps the un-gating mechanism; the polyfill logic is unchanged.
A single-shot live check under the H1 config confirms the new mechanism un-gates the
polyfill: a ~60K-token `/v1/messages` request returns
`context_management.applied_edits[0]` = `compact_20260112` with a real summarization
sub-call (`summary_input_tokens: 60116`, `summary_output_tokens: 86`) — the inverse of
global-`true`-alone, which strips `context_management` to null. This is the behavior
claude's own auto-compact cannot provide (one fire per session).

**Known limitation — summarization sub-call headroom.** The polyfill's summarization
sub-call requests a `max_tokens` of `context_management_summary_max_tokens` (default
4096) **on top of** the full history. So the history at fire time must satisfy
`history + summary_max_tokens + prompt < --max-model-len` (128K MoE / 64K dense). The
default thresholds (90000 / 50000) leave enough headroom for **gradual** per-turn growth
(90K + 4096 + prompt ≈ 94K < 128K). A single turn that jumps context to >~123K (MoE) /
>~61.5K (dense) overflows the summary call → the polyfill fails → the main call 400s.
This is an inherent polyfill limitation, not a code bug; it does not affect the normal
growth pattern where context crosses the threshold incrementally. The dense stack
overrides `context_management_summary_max_tokens: 2048` (vs the 4096 default) to push
its overflow point up by ~2K (50K + 2048 vs 50K + 4096) for the single-big-turn edge
case; MoE keeps the 4096 default (34K+ of margin).

**TODO — verify the fix in claude code.** To confirm repeated proxy-side compaction
works end-to-end on this stack, run the checked-in probe (an integration test, skipped
by default even under `make test-integration`; env-gated, ~10–15 min, needs the stack up
with `CLAUDE_QWEN_PROXY_COMPACT=1`):

```
RUN_LIVE_COMPACTION_PROBE=1 python3 -m pytest tests/integration/test_compaction_probe.py -m integration -o addopts="" -s
```

This is a **temporary layer** intended to be removed once the upstream claude-code
gateway-mode compaction blockers are lifted — notably
[#65585](https://github.com/anthropics/claude-code/issues/65585) (auto-compact gated
behind first-party auth since v2.1.161),
[#68522](https://github.com/anthropics/claude-code/issues/68522) (custom
`ANTHROPIC_BASE_URL` models assumed 200k, no way to declare a larger window), and
[#44354](https://github.com/anthropics/claude-code/issues/44354) (reactive compaction
handler null on resumed sessions — the per-session breaker this polyfill bypasses).
Until then it is the only way to get repeated compaction on this proxy.

---

## Part 4 — Scaling Beyond One A10

### PCIe bandwidth — grounding the TP=2-on-x4 question

`make bench_pcie` (`scripts/pcie_bw_bench.py`, run inside the vLLM image so the host needs no torch) measures GPU↔host cudaMemcpy
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
1.22 GiB for a 2000-token prefill.

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

### Two A10 cards — TP=2 projections for Dense and MoE

Two A10s on a single host (no NVLink — x4 only): lifts context ceilings and removes
the MoE offload tax. The PCIe grounding above shows **TP=2 pays off for decode**
(all-reduce <0.5 % of the step); prefill pays ~10–20 %.

> Validation status: the second A10 hardware is on the way in. The TP=2 setup is
> planned to be validated on real hardware in the next few weeks; until then,
> all 2×A10 rows below are projections, not measured results.

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
                   27B decode   ~640 KiB/token  → <0.5% of 47 ms step
                   35B decode   ~320 KiB/token  → <1–2% of 6.7 ms step
                   27B prefill  ~1.22 GiB / 2000 → ~10–20% of step
                 transport: P2P (1 hop) if ACS allows, else SHM (2 hops)
```

#### Footprint: MTP + visual are downloaded, not loaded

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

#### Why TP=2, not PP=2, for single-user serving on x4

- **TP=2 parallelizes the GDDR6 weight reads** (the actual decode bottleneck) —
  each GPU reads half the weights per step → faster decode. All-reduce at
  batch=1 is small on x4: ~640 KiB/token one-way for the 27B dense and
  ~320 KiB/token for the 35B MoE nominal two-collective/layer path used below.
- **PP=2 at batch=1 does not overlap** (no pipeline to fill) — stages run
  serially, one GPU idle each half-step → ~same latency as a single GPU that
  could fit the model, *plus* comm. PP makes a too-big model **fit** but does
  not make it **faster**. PP's smaller comm footprint only wins at **high
  batch / multi-user throughput**, a different goal.

#### Projected decode tps

Dense decode is GDDR6-bound: `tps ≈ BW / weights_read_per_token × ~0.7`. The formula
is decode-only; MoE uses a separate active-read envelope because the single-GPU
MoE measurement is contaminated by UVA offload stalls. 1×A10 rows are measured;
2×A10 rows are untested projections (no second GPU available for measurement).

| model | setup | offload | decode tps | prefill tps | context | fits 2×A10? |
|---|---|---|---|---|---|---|
| 27B Dense | 1× A10 | none | ~21 | ~1026 | 64k | n/a |
| 35B MoE | 1× A10 | 2.2 GiB | ~15.4 | ~966→698 | 128k | n/a |
| 27B Dense | 2× A10 TP=2 | none | ~35 | not modeled | 128k (proj; VRAM ceiling ~640k tok) | yes |
| 35B MoE | 2× A10 TP=2 | none | ~80-150 | not modeled | 256k (proj) | yes (proj) |

The 2×A10 projections' reasoning:

- **Dense 27B TP=2 fits 128k with large margin.** Weights shard to ~9.4 GiB/GPU;
  KV+state is ~35 KiB/token, halved to ~17 KiB/GPU under TP=2, so 128k costs
  ~2.2 GiB/GPU. With a conservative runtime reserve, that leaves ~8–9 GiB/GPU
  free; the VRAM ceiling is roughly ~640k tokens, so 128k is limited by
  `--max-model-len`, not memory. The single-A10 dense path needs
  `--cpu-offload-gb 6` for 128k and collapses to ~2.5 tps; TP=2 avoids that
  offload path.
- **Dense 27B decode at 128k is ~35 tps projected.** At 128k, the 16 fp8
  full-attention layers add ~2.0 GiB/GPU of KV read per decoded token on top of
  the ~9.4 GiB/GPU weight shard. Applying the measured dense efficiency gives
  ~34–36 tps, so use ~35 tps as the 128k planning number.

- **No offload needed** — ~21.5 GiB / 2 ≈ 11 GiB/GPU weights, ample room for
  KV+state. Removing offload removes the PCIe-stall tax (the 80 %-power ceiling)
  and the 100 %-CPU offloader thread.
- **Decode planning estimate is context-sensitive: ~120–180 tps at short/moderate
  context, degrading toward ~80–150 tps at 256k.** Do not anchor the no-offload
  TP=2 case to the measured 15.4 tps single-A10 row: that row is the 2.2 GiB UVA
  offload path, so it includes PCIe weight-read stalls and the 100 %-CPU offloader
  thread. A no-offload TP=2 deployment moves the active weights back into GDDR6
  and shards the language weights across both cards.
- **GDDR6 bandwidth does not rule out ~150 tps, but the raw ceiling falls with
  context.** A10 GDDR6 is ~600 GB/s (~559 GiB/s). The MoE active weight read is
  roughly ~1.4–1.5 GiB/token total, or ~0.7–0.75 GiB/token per GPU under TP=2,
  so the weights-only near-zero-context ceiling is ~745–800 tps before
  kernel/runtime overhead. At long context the 10 full-attention layers add KV
  reads: roughly ~0.6 GiB/GPU at 128k and ~1.3 GiB/GPU at 256k, lowering the
  GDDR6 ceiling to about ~400 tps at 128k and ~275 tps at 256k before fixed
  GatedDeltaNet state reads and runtime overhead. A ~150 tps target is therefore
  light at short context (~20 % of the weights-only ceiling) but material at
  256k (~55 % of the KV-inclusive ceiling).
- **PCIe x4 all-reduce is a tax, but not a 20-tps limiter for decode.** The MoE
  hidden size is 2048 with 40 layers, so a two-collective/layer decode path is
  ~320 KiB/token. That is ~50 µs over P2P and ~100 µs through SHM fallback at
  the measured ~6.6 GB/s x4 bandwidth. At ~150 tps (6.7 ms/token),
  bandwidth-only comm is <1–2 %. Collective launch latency may be larger than
  the transfer time, especially with `--enforce-eager`, which is why the
  projection is far below the raw GDDR6 ceiling; it still should not collapse to
  the offloaded 15.4 tps regime if TP is actually active.
- **Context to 256k (conservative — VRAM is not the limiter)** — ~10 GiB/GPU
  for KV+state is far more than 256k needs: scaling the measured 1× ratio
  (1.78 GiB → 150 349 tokens) by the halved per-GPU KV gives ~1.7 M tokens of
  capacity per GPU. 256k is therefore bounded by the model's `--max-model-len` /
  max-context config, not by VRAM; the 1072-token mamba align block still
  requires `--max-num-batched-tokens ≥ 1072`.

#### Config diff (single-GPU → 2-GPU TP=2)

| setting | 27B single A10 | 27B 2× TP=2 | 35B MoE single A10 | 35B MoE 2× TP=2 |
|---|---|---|---|---|
| `--tensor-parallel-size` | 1 | 2 | 1 | 2 |
| `gpus` | all (1) | all (2) | all (1) | all (2) |
| `--gpu-memory-utilization` | 0.97 | 0.97 | 0.95 | 0.95–0.97 (more headroom) |
| `--cpu-offload-gb` | — | — | **2.2** | **drop (no offload needed)** |
| `--max-model-len` | 64000 | **128000 (proj)** | 128000 | **256000 (proj)** |
| `--max-num-batched-tokens` | 1024 | 1024 | 1280 | 1280 (≥1072 align) |
| `--max-num-seqs` | 1 | 1 | 1 | 1 |
| weights / GPU | 18.83 GiB | ~9.4 GiB | ~19.2 GiB (+2.2 off) | ~10.75 GiB |
| `shm_size` | 32g | 32g | 32g | 32g |
| `NCCL_DEBUG` env | — | `INFO` | `INFO` | `INFO` |

#### How to verify when running it for real

- `nvidia-smi` per-GPU memory **balanced** (~equal) → TP shards both layer types.
- `NCCL_DEBUG=INFO` log: `via P2P/IPC` (best) or `via SHM` (fallback, fine);
  `via NET` would indicate TCP (shouldn't happen intra-node).
- 27B dense decode ~35 tps at 128k; 35B MoE decode ~120+ tps at short/moderate context, or
  materially above 15.4 tps at 256k → no-offload TP is working. MoE stuck near
  15.4 tps means the run is still effectively on the offloaded/single-GPU path,
  TP is not sharding the hybrid layers, or init fell back/failed.

#### Open risks (not resolvable on a 1-GPU box)

- **vLLM's TP sharding of the GatedDeltaNet linear-attn layers is unverified**
  — the main risk for both models. If TP only shards the full-attn layers, load
  is imbalanced and the 2× never materializes. The balanced-memory check above
  catches this.
- x4 prefill tax ~10–20 % (P2P vs SHM).
- PP=2 at batch=1 → ~same latency as 1 GPU + comm; don't use PP for single-user.
