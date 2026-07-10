.DEFAULT_GOAL := help

PY      ?= python3
BENCH   ?= scripts/coding_session_bench.py
TURNS   ?= 27  # reaches ~64k seq (full context): ~2k input + ~500 output/turn

.PHONY: help run start stop run35 start35 stop35 clean test test-integration test-all bench bench35 bench_pcie idle-test litellm-logs litellm-logs35

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run: ## Start the stack in the foreground (console, Ctrl-C to stop)
	docker compose up

start: ## Start the stack detached (background)
	docker compose up -d

stop: ## Stop the stack (containers kept, not removed)
	docker compose stop

run35: ## Start the 35B MoE stack in the foreground (Ctrl-C to stop)
	docker compose -f docker-compose.moe.yaml up

start35: ## Start the 35B MoE stack detached (background)
	docker compose -f docker-compose.moe.yaml up -d

stop35: ## Stop the 35B MoE stack (containers kept, not removed)
	docker compose -f docker-compose.moe.yaml stop

clean: ## FULL RESET (destructive): wipe Open WebUI DB + vLLM caches across BOTH stacks
	@echo "==> Destroying volumes (IRREVERSIBLE): open-webui-data, vllm_cache, vllm_cache_moe"
	@echo "    open-webui-data holds your admin account + chat history -> gone."
	-docker compose down -v
	-docker compose -f docker-compose.moe.yaml down -v
	@echo "==> Done. Recreate with: make start  (27B)  or  make start35  (35B MoE)"
	@echo "    First visit to :3000 re-creates the Open WebUI admin account; the model list"
	@echo "    re-seeds from OPENAI_API_BASE_URL (litellm:4000/v1)."

test: ## Fast unit + config tests with coverage (no GPU/docker)
	$(PY) -m pytest

test-integration: ## Long integration tests against the LIVE stack (run make start35 first)
	$(PY) -m pytest tests/integration -m integration -o addopts="" -s

test-all: ## Run unit + integration (integration skipped unless the stack is up)
	$(PY) -m pytest tests

bench: ## Run the growing coding-session bench (override with TURNS=N)
	$(PY) $(BENCH) --turns $(TURNS)

bench35: ## Bench the 35B MoE stack via vLLM (qwen3.6-35b-a3b; override with TURNS=N)
	$(PY) $(BENCH) --model qwen3.6-35b-a3b --turns $(TURNS)

bench_pcie: ## Measure GPU<->host PCIe bandwidth (free GPU needed: `make stop` first)
	$(PY) scripts/pcie_bw_bench.py

idle-test: ## (read-only) measure idle/stop/cold-wake power — see PLAN Phase 0
	@echo "Phase 0 power test (read-only; expect ~15W stopped vs ~50-75W served):"
	@echo "  0. make stop && make stop35   (clean GPU)"
	@echo "  1. served idle : make start; wait /health 200 + 120s settle"
	@echo "  2. stop floor  : make stop; wait 120s"
	@echo "  3. cold wake   : time make start; poll curl http://localhost:4000/health until 200"
	@echo "  power sampler : nvidia-smi -q -d POWER | grep -E 'Instantaneous|Average'"

litellm-logs: ## Tail the LiteLLM proxy (27B stack: translate + wake/idle lifecycle)
	docker compose logs -f litellm

litellm-logs35: ## Tail the LiteLLM proxy (35B MoE stack)
	docker compose -f docker-compose.moe.yaml logs -f litellm