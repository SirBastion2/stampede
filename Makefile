.PHONY: setup lint test ci determinism-smoke arch-doc-check ui-build

# PR base for gate §10.3 #9 (override in CI with the merge base SHA).
CI_BASE ?= origin/main

setup:
	python3 -m pip install -e ".[dev,server]"

lint:
	python3 -m ruff check .
	python3 -m mypy contracts engine mocks

test:
	python3 -m pytest -q

# ui/crowd compiles ../market/src, so ui/market needs its node_modules first (E4).
ui-build:
	cd ui/market && npm ci
	cd ui/crowd && npm ci && npm run build
	cd ui/market && npm run build

determinism-smoke:
	python3 scripts/determinism_smoke.py

arch-doc-check:
	python3 scripts/check_architecture_doc.py $(CI_BASE)

ci:
	@echo "=== §10.3 gate 1: lint + mypy ==="
	@$(MAKE) lint
	@echo "=== §10.3 gates 2–3: unit + contract tests ==="
	@$(MAKE) test
	@echo "=== §10.3 gate 9: architecture doc (base $(CI_BASE)) ==="
	@$(MAKE) arch-doc-check
	@echo "=== determinism smoke (fixed-seed fake engine) ==="
	@$(MAKE) determinism-smoke
	@echo "=== make ci: all CPU gates passed ==="
