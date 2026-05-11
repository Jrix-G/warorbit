#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE

GAMES="${GAMES:-16}"
WORKERS="${WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-220}"
WEIGHTS="${WEIGHTS:-evaluations/scorer_v14.npz}"
OPPONENTS=(
  notebook_orbitbotnext
  notebook_distance_prioritized
  notebook_physics_accurate
  notebook_pascalledesma_orbitwork_v14
)

run_variant() {
  local agent="$1"
  local profile="$2"
  echo
  echo "=== V14_4P_AGENT=${agent} V14_4P_PROFILE=${profile} games=${GAMES} ==="
  V14_4P_AGENT="${agent}" V14_4P_PROFILE="${profile}" \
    python3 -u benchmark_v14.py \
      --v14-weights "${WEIGHTS}" \
      --games "${GAMES}" \
      --workers "${WORKERS}" \
      --max-steps "${MAX_STEPS}" \
      --modes 4p \
      --bots v14 \
      --opponents "${OPPONENTS[@]}"
}

run_variant distance eco
run_variant orbitbotnext eco
run_variant distance closer
run_variant orbitbotnext closer
run_variant distance base
run_variant physics eco
