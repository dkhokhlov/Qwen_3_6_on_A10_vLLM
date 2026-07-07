.DEFAULT_GOAL := help

PY      ?= python3
BENCH   ?= scripts/coding_session_bench.py
TURNS   ?= 27  # reaches ~64k seq (full context): ~2k input + ~500 output/turn

.PHONY: help run start stop run35 start35 stop35 bench bench35 bench_pcie

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

bench: ## Run the growing coding-session bench (override with TURNS=N)
	$(PY) $(BENCH) --turns $(TURNS)

bench35: ## Bench the 35B MoE on vllm-qwen35b (override with TURNS=N)
	$(PY) $(BENCH) --model qwen3.6-35b-a3b --turns $(TURNS)

bench_pcie: ## Measure GPU<->host PCIe bandwidth (free GPU needed: `make stop` first)
	$(PY) scripts/pcie_bw_bench.py