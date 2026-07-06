.DEFAULT_GOAL := help

PY      ?= python3
BENCH   ?= scripts/coding_session_bench.py
TURNS   ?= 8

.PHONY: help run profile

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run: ## Start the vLLM + open-webui stack (detached)
	docker compose up -d

profile: ## Run the growing coding-session bench (override with TURNS=N)
	$(PY) $(BENCH) --turns $(TURNS)