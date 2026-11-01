# FairPath -- one-command workflows. Run `make help` to see everything.
PY ?= python
PIP ?= $(PY) -m pip

.DEFAULT_GOAL := help
.PHONY: help setup data synthetic reproduce train train-fast report test test-all lint format api web web-dev run docker docker-run clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## install the package with API + dev extras and git hooks
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[api,dev]" --no-deps
	pre-commit install || true

data:  ## download the two UCI datasets into data/raw (not committed)
	fairpath download

synthetic:  ## regenerate the synthetic look-alike data from the real data
	fairpath synthesize

reproduce:  ## re-run the original thesis protocol and compare with printed numbers
	fairpath reproduce-thesis

train:  ## full pipeline: tune, compare, mitigation study, final models, reports (~25 min on 2 cores)
	fairpath all

train-fast:  ## same pipeline on the SYNTHETIC data with tuning skipped (a few minutes)
	fairpath all --source synthetic --no-tune

report:  ## rebuild reports/overview.json, docs/figures, docs/RESULTS.md and the dashboard's static data
	fairpath report
	$(PY) scripts/export_static_data.py

test:  ## fast unit + integration tests
	pytest -m "not slow" --cov --cov-report=term-missing:skip-covered

test-all:  ## every test, including the end-to-end pipeline smoke test
	pytest --cov

lint:  ## ruff + black check
	ruff check src tests
	black --check src tests

format:  ## auto-format
	ruff check --fix src tests
	black src tests

api:  ## run the API on :8000 (serves the built dashboard too, if web/out exists)
	uvicorn fairpath.api.main:app --reload --port 8000

web:  ## build the dashboard into web/out
	cd web && npm ci && npm run build

web-dev:  ## dashboard dev server on :3000 (proxies /api to :8000)
	cd web && npm run dev

run: web  ## build the dashboard and serve everything on :8000
	uvicorn fairpath.api.main:app --port 8000

docker:  ## build the production image
	docker build -t fairpath:latest .

docker-run:  ## run the production image on :7860
	docker run --rm -p 7860:7860 fairpath:latest

clean:  ## remove caches and build output
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov web/.next web/out
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
