# The 14B model id; point LLM_MODEL at a local MLX folder to avoid a download.
LLM_MODEL ?= mlx-community/Qwen3-14B-4bit
GPU_LOCK ?=
RUN = uv run python -m

.PHONY: data bench llm charts test lint all

# Download SEC + Wikidata names (slow: ~8k SEC requests at <10/s) and regenerate the splits.
data:
	$(RUN) bench.fetch_sec
	$(RUN) bench.fetch_wikidata
	$(RUN) bench.generate --scale

# Everything that runs on the CPU: blocking, scorers (trains the shipped model), scaling.
# generate is deterministic: it rewrites dev/test byte for byte and adds the 50k scale.csv (not committed).
bench:
	$(RUN) bench.generate --scale
	$(RUN) bench.run_blocking
	$(RUN) bench.run_scorers
	$(RUN) bench.run_scaling

# Every LLM judge run (Apple Silicon, a few GPU-hours for the 14B). Resumable, in chunks.
llm:
	LLM_MODEL=$(LLM_MODEL) GPU_LOCK="$(GPU_LOCK)" bash bench/run_llm_all.sh

# Calibrate the judge on dev, combine everything on disk into results/cascade.json, redraw docs/*.png.
charts:
	$(RUN) bench.calibrate
	$(RUN) bench.run_cascade
	$(RUN) bench.charts

test:
	uv run pytest -q

lint:
	uv run ruff check .

all: bench llm charts
