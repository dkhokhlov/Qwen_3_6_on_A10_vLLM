.DEFAULT_GOAL := help

PY      ?= python3
VENV    := .venv
UV      ?= uv
BENCH   ?= scripts/coding_session_bench.py
TURNS   ?= 27  # reaches ~64k seq (full context): ~2k input + ~500 output/turn
VLLM_IMAGE := $(shell grep -m1 'image: vllm/vllm-openai' docker-compose.yaml | sed 's/.*image: //; s/ .*//')

.PHONY: help run start stop run35 start35 stop35 run-tp2 start-tp2 stop-tp2 run35-tp2 start35-tp2 stop35-tp2 stop-all clean ci test test-integration test-all bench bench35 bench-tp2 bench35-tp2 bench_pcie test-pcie idle-test litellm-logs litellm-logs35 litellm-logs-tp2 litellm-logs35-tp2

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run: ## Start the 27B dense stack in the foreground (Ctrl-C to stop)
	docker compose -f docker-compose.yaml up

start: ## Start the 27B dense stack detached (background)
	docker compose -f docker-compose.yaml up -d

stop: ## Stop the 27B dense stack (containers kept, not removed)
	docker compose -f docker-compose.yaml stop

run35: ## Start the 35B MoE stack in the foreground (Ctrl-C to stop)
	docker compose -f docker-compose.moe.yaml up

start35: ## Start the 35B MoE stack detached (background)
	docker compose -f docker-compose.moe.yaml up -d

stop35: ## Stop the 35B MoE stack (containers kept, not removed)
	docker compose -f docker-compose.moe.yaml stop

run-tp2: ## Start the 27B dense TP=2 stack in the foreground, sharded across BOTH A10s (Ctrl-C to stop)
	docker compose -f docker-compose.tp2.yaml up

start-tp2: ## Start the 27B dense TP=2 stack detached, sharded across BOTH A10s (256k ctx)
	docker compose -f docker-compose.tp2.yaml up -d

stop-tp2: ## Stop the 27B dense TP=2 stack (containers kept, not removed)
	docker compose -f docker-compose.tp2.yaml stop

run35-tp2: ## Start the 35B MoE TP=2 stack in the foreground, sharded across BOTH A10s (Ctrl-C to stop)
	docker compose -f docker-compose.moe.tp2.yaml up

start35-tp2: ## Start the 35B MoE TP=2 stack detached, sharded across BOTH A10s (256k ctx, NO offload)
	docker compose -f docker-compose.moe.tp2.yaml up -d

stop35-tp2: ## Stop the 35B MoE TP=2 stack (containers kept, not removed)
	docker compose -f docker-compose.moe.tp2.yaml stop

stop-all: stop stop35 stop-tp2 stop35-tp2 ## Stop ALL stacks (dense + MoE, 1x + TP=2; containers kept, not removed)

clean: ## FULL RESET (destructive): wipe Open WebUI DB + vLLM caches across BOTH stacks
	@echo "==> Destroying volumes (IRREVERSIBLE): open-webui-data, vllm_cache, vllm_cache_moe"
	@echo "    open-webui-data holds your admin account + chat history -> gone."
	-docker compose -f docker-compose.yaml down -v
	-docker compose -f docker-compose.moe.yaml down -v
	-docker compose -f docker-compose.tp2.yaml down -v
	-docker compose -f docker-compose.moe.tp2.yaml down -v
	@echo "==> Done. Recreate with: make start  (27B)  or  make start35  (35B MoE)"
	@echo "    First visit to :3000 re-creates the Open WebUI admin account; the model list"
	@echo "    re-seeds from OPENAI_API_BASE_URL (litellm:4000/v1)."

ci: ## Bootstrap the project venv (.venv via uv, python 3.12) with the test deps; test targets depend on it
	@[ -x $(VENV)/bin/python ] || $(UV) venv --python 3.12 $(VENV)
	@$(UV) pip install --python $(VENV)/bin/python -r requirements-test.txt

test: ci ## Fast unit + config tests with coverage (no GPU/docker)
	$(VENV)/bin/python -m pytest

test-integration: ci ## Long integration tests against the LIVE stack (run make start35 first)
	$(VENV)/bin/python -m pytest tests/integration -m integration -o addopts="" -s

test-all: ci ## Run unit + integration (integration skipped unless the stack is up; gpu tier is make test-pcie)
	$(VENV)/bin/python -m pytest tests/unit tests/integration

bench: ## Run the growing coding-session bench (override with TURNS=N)
	$(PY) $(BENCH) --turns $(TURNS)

bench35: ## Bench the 35B MoE stack via vLLM (qwen3.6-35b-a3b; override with TURNS=N)
	$(PY) $(BENCH) --model qwen3.6-35b-a3b --turns $(TURNS)

bench-tp2: ## Bench the 27B dense TP=2 stack via vLLM (qwen3.6-27b; override with TURNS=N)
	$(PY) $(BENCH) --model qwen3.6-27b --turns $(TURNS)

bench35-tp2: ## Bench the 35B MoE TP=2 stack via vLLM (qwen3.6-35b-a3b; override with TURNS=N)
	$(PY) $(BENCH) --model qwen3.6-35b-a3b --turns $(TURNS)

bench_pcie: ## Measure GPU<->host PCIe bandwidth in the vLLM image (free GPU: `make stop` first)
	$(if $(VLLM_IMAGE),,$(error VLLM_IMAGE not found: expected an 'image: vllm/vllm-openai' line in docker-compose.yaml))
	docker run --rm --gpus all --entrypoint python3 -v "$(CURDIR)":/repo:ro -w /repo \
		$(VLLM_IMAGE) scripts/pcie_bw_bench.py

test-pcie: ## Run the pcie unit tests INSIDE the vLLM image (needs GPU; fails without one)
	$(if $(VLLM_IMAGE),,$(error VLLM_IMAGE not found: expected an 'image: vllm/vllm-openai' line in docker-compose.yaml))
	docker run --rm --gpus all --entrypoint bash -v "$(CURDIR)":/repo:ro -w /repo \
		$(VLLM_IMAGE) -c 'pip install -q pytest pyyaml && python3 -m pytest tests/gpu/test_pcie.py -o addopts="" -p no:cacheprovider'

idle-test: ## (read-only) measure idle/stop/cold-wake power — see PLAN Phase 0
	@echo "Phase 0 power test (read-only; expect ~15W stopped vs ~50-75W served):"
	@echo "  0. make stop && make stop35   (clean GPU)"
	@echo "  1. served idle : make start; wait /health 200 + 120s settle"
	@echo "  2. stop floor  : make stop; wait 120s"
	@echo "  3. cold wake   : time make start; poll curl http://localhost:4000/health until 200"
	@echo "  power sampler : nvidia-smi -q -d POWER | grep -E 'Instantaneous|Average'"

litellm-logs: ## Tail the LiteLLM proxy (27B stack: translate + wake/idle lifecycle)
	docker compose -f docker-compose.yaml logs -f litellm

litellm-logs35: ## Tail the LiteLLM proxy (35B MoE stack)
	docker compose -f docker-compose.moe.yaml logs -f litellm

litellm-logs-tp2: ## Tail the LiteLLM proxy (27B dense TP=2 stack; shows NCCL transport in vllm logs too)
	docker compose -f docker-compose.tp2.yaml logs -f litellm

litellm-logs35-tp2: ## Tail the LiteLLM proxy (35B MoE TP=2 stack; shows NCCL transport in vllm logs too)
	docker compose -f docker-compose.moe.tp2.yaml logs -f litellm