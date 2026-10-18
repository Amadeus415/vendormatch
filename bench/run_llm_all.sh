#!/usr/bin/env bash
# Every LLM run behind the README numbers, in the order they were run.
# Each call is resumable and capped with --limit, so a shared GPU can be released between chunks:
#   GPU_LOCK="lockf -k /path/gpu.lock" LLM_MODEL=/path/to/Qwen3-14B-4bit bash bench/run_llm_all.sh
set -euo pipefail

MODEL=${LLM_MODEL:-mlx-community/Qwen3-14B-4bit}
ID=mlx-community/Qwen3-14B-4bit
LOCK=${GPU_LOCK:-}
HEAD=${HEAD_PAIRS:-3000}      # most-uncertain test pairs judged exhaustively
TAIL=${TAIL_SAMPLE:-800}      # uniform sample of the rest, for the "LLM on everything" estimate
run() { $LOCK uv run python -m bench.run_llm "$@"; }
until_done() {  # repeat a resumable run in chunks until it reports nothing left to do
  local out
  while true; do
    out=$(run "$@" --limit "${CHUNK:-600}" 2>&1 | tee /dev/stderr)
    grep -q "judging [1-9]" <<<"$out" || break
  done
}

# 0. Prompt development, on dev pairs only: the first draft (v1) against the revision (v2). Run once;
#    the samples on disk were drawn before the dev folds were grouped by company, so they are not redrawn.
[ -f results/llm/dev-prompt-v1.jsonl ] || until_done --tag dev-prompt-v1 --split dev --pairs band --sample 300 --no-reason --prompt v1 --model "$MODEL" --model-id $ID
[ -f results/llm/dev-prompt-v2.jsonl ] || until_done --tag dev-prompt-v2 --split dev --pairs band --sample 300 --no-reason --prompt v2 --model "$MODEL" --model-id $ID

# 1. The cascade: Qwen3-14B decides every pair in the uncertain band (decision token only).
until_done --tag qwen3-14b_band --pairs band --no-reason --model "$MODEL" --model-id $ID

# 2. The same judge with a written reason, on 150 random band pairs: examples and the latency of reasons.
until_done --tag qwen3-14b_reasons --pairs band --sample 150 --model "$MODEL" --model-id $ID

# 3. The cost/accuracy curve: most uncertain pairs first, reusing the band verdicts.
until_done --tag qwen3-14b_all --pairs all --sample "$HEAD" --no-reason --reuse qwen3-14b_band --model "$MODEL" \
  --model-id $ID

# 4. A random sample of the remaining pairs, to estimate "LLM on every pair".
until_done --tag qwen3-14b_tail --pairs tail --head "$HEAD" --sample "$TAIL" --no-reason --model "$MODEL" --model-id $ID

# 5. Thinking on, with a 256-token budget, on 100 random band pairs.
until_done --tag qwen3-14b_think256 --pairs band --sample 100 --no-reason --think 256 --model "$MODEL" --model-id $ID

# 6. Smaller judges on the same band pairs. Override with local folders: SMALL_MODELS="qwen3-1.7b=/path/to/model"
SMALL=${SMALL_MODELS:-"qwen3-0.6b=mlx-community/Qwen3-0.6B-4bit qwen3-1.7b=mlx-community/Qwen3-1.7B-4bit"}
for entry in $SMALL; do
  tag=${entry%%=*}
  path=${entry#*=}
  until_done --tag "${tag}_band" --pairs band --no-reason --model "$path" --model-id "mlx-community/$(basename "$path")"
done

# 7. Every dev pair in the band (v2 prompt), to calibrate the judge's yes/no into probabilities for clustering.
until_done --tag dev-band-v2 --split dev --pairs band --no-reason --reuse dev-prompt-v2 --model "$MODEL" --model-id $ID
