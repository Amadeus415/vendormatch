#!/usr/bin/env bash
# Every LLM run behind the README numbers, in the order they were run.
# Each call is resumable and capped with --limit, so a shared GPU can be released between chunks:
#   GPU_LOCK="lockf -k /path/gpu.lock" LLM_MODEL=/path/to/Qwen3-14B-4bit bench/run_llm_all.sh
set -euo pipefail

MODEL=${LLM_MODEL:-mlx-community/Qwen3-14B-4bit}
LOCK=${GPU_LOCK:-}
HEAD=${HEAD_PAIRS:-4000}      # most-uncertain pairs judged exhaustively
TAIL=${TAIL_SAMPLE:-1200}     # uniform sample of the rest, for the "LLM on everything" estimate
run() { $LOCK uv run python -m bench.run_llm "$@"; }
until_done() {  # repeat a resumable run in chunks until it reports nothing left to do
  while run "$@" --limit "${CHUNK:-1500}" | tee /dev/stderr | grep -q "judging [1-9]"; do :; done
}

# 1. The cascade itself: Qwen3-14B on the uncertain band, one pair at a time, with a reason.
until_done --tag qwen3-14b_band --pairs band --model "$MODEL" --model-id mlx-community/Qwen3-14B-4bit

# 2. The cost/accuracy curve: decision only, most uncertain pairs first (band pairs are reused, not rerun).
until_done --tag qwen3-14b_all --pairs all --sample "$HEAD" --no-reason --reuse qwen3-14b_band --model "$MODEL" \
  --model-id mlx-community/Qwen3-14B-4bit

# 3. A random sample of the remaining pairs, to estimate "LLM on every pair".
until_done --tag qwen3-14b_tail --pairs tail --head "$HEAD" --sample "$TAIL" --no-reason --model "$MODEL" \
  --model-id mlx-community/Qwen3-14B-4bit

# 4. Thinking mode with a 256-token budget on a random 150 band pairs.
until_done --tag qwen3-14b_think256 --pairs band --sample 150 --think 256 --model "$MODEL" \
  --model-id mlx-community/Qwen3-14B-4bit

# 5. Smaller judges on the same band pairs.
for m in Qwen3-0.6B-4bit Qwen3-1.7B-4bit Qwen3-4B-4bit; do
  tag=$(echo "$m" | tr 'A-Z' 'a-z' | sed 's/-4bit//')
  until_done --tag "${tag}_band" --pairs band --model "mlx-community/$m" --model-id "mlx-community/$m" || true
done
