#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-v9_kaggle_pool15_90m}"
SEED_CHECKPOINT="${SEED_CHECKPOINT:-evaluations/v9_kaggle_30m_best.npz}"
LATEST="evaluations/${RUN_NAME}_latest.npz"
BEST="evaluations/${RUN_NAME}_best.npz"
POLICY="evaluations/${RUN_NAME}_policy.npz"
LOG="evaluations/${RUN_NAME}_train.jsonl"

mkdir -p evaluations

if [[ ! -f "$SEED_CHECKPOINT" ]]; then
  echo "Seed checkpoint missing: $SEED_CHECKPOINT" >&2
  exit 1
fi

if [[ "${FRESH:-0}" == "1" ]]; then
  rm -f "$LATEST" "$BEST" "$POLICY" "$LOG"
fi

# Resume from the validated 30m Kaggle best. Copy both latest and best so the
# trainer can export best on timeout instead of falling back to a degraded latest.
if [[ ! -f "$LATEST" ]]; then
  cp "$SEED_CHECKPOINT" "$LATEST"
fi
if [[ ! -f "$BEST" ]]; then
  cp "$SEED_CHECKPOINT" "$BEST"
fi

TRAIN_OPPONENTS=(
  random
  noisy_greedy
  starter
  distance
  sun_dodge
  structured
  orbit_stars
  bot_v7
  notebook_tactical_heuristic
  notebook_mdmahfuzsumon_how_my_ai_wins_space_wars
  notebook_sigmaborov_orbit_wars_2026_starter
  notebook_sigmaborov_orbit_wars_2026_tactical_heuristic
  notebook_orbitbotnext
  notebook_distance_prioritized
  notebook_physics_accurate
)

EVAL_OPPONENTS=(
  heldout_random
  heldout_greedy
  notebook_johnjanson_lb_max_score_1000_agi_is_here
  notebook_romantamrazov_orbit_star_wars_lb_max_1224
  notebook_pascalledesma_orbitbotnext
  notebook_sigmaborov_lb_958_1_orbit_wars_2026_reinforce
  notebook_ykhnkf_distance_prioritized_agent_lb_max_score_1100
  notebook_sigmaborov_lb_928_7_physics_accurate_planner
)

BENCHMARK_OPPONENTS=(
  notebook_orbitbotnext
  notebook_distance_prioritized
  notebook_physics_accurate
  notebook_djenkivanov_orbit_wars_optimized_nearest_planet_sniper
  notebook_pascalledesma_orbitwork_v14
  notebook_johnjanson_lb_max_score_1000_agi_is_here
  notebook_romantamrazov_orbit_star_wars_lb_max_1224
  notebook_pascalledesma_orbitbotnext
  notebook_sigmaborov_lb_958_1_orbit_wars_2026_reinforce
  notebook_ykhnkf_distance_prioritized_agent_lb_max_score_1100
  notebook_sigmaborov_lb_928_7_physics_accurate_planner
  notebook_debugendless_orbit_wars_sun_dodging_baseline
  notebook_mdmahfuzsumon_how_my_ai_wins_space_wars
  notebook_sigmaborov_orbit_wars_2026_starter
  notebook_sigmaborov_orbit_wars_2026_tactical_heuristic
)

echo "Run: $RUN_NAME"
echo "Seed: $SEED_CHECKPOINT"
echo "Latest: $LATEST"
echo "Best: $BEST"
echo "Policy: $POLICY"
echo "Log: $LOG"

python3 run_v9.py \
  --game-engine kaggle \
  --minutes 90 \
  --hard-timeout-minutes 90 \
  --workers 8 \
  --pairs 3 \
  --games-per-eval 1 \
  --eval-games 16 \
  --benchmark-games 32 \
  --min-promotion-benchmark-games 32 \
  --benchmark-progress-every 2 \
  --eval-every 1 \
  --benchmark-every 1 \
  --max-steps 220 \
  --eval-max-steps 220 \
  --four-player-ratio 0.80 \
  --eval-four-player-ratio 0.80 \
  --benchmark-four-player-ratio 0.80 \
  --train-search-width 3 \
  --train-simulation-depth 0 \
  --train-simulation-rollouts 0 \
  --train-opponent-samples 1 \
  --search-width 4 \
  --simulation-depth 0 \
  --simulation-rollouts 0 \
  --pool-limit 15 \
  --min-benchmark-score 0.24 \
  --guardian-min-benchmark-4p 0.24 \
  --guardian-min-benchmark-backbone 0.06 \
  --guardian-max-benchmark-fronts 3.45 \
  --guardian-max-generalization-gap 0.42 \
  --max-generalization-gap 0.42 \
  --front-lock-turns 12 \
  --max-focus-targets-4p 4 \
  --target-active-fronts 2.25 \
  --front-penalty-weight 0.042 \
  --front-pressure-attack-penalty 0.055 \
  --front-pressure-plan-bias 0.085 \
  --backbone-penalty-weight 0.080 \
  --backbone-bonus-weight 0.090 \
  --target-backbone-turn-frac 0.14 \
  --candidate-diversity 1.35 \
  --exploration-rate 0.06 \
  --reward-noise 0.006 \
  --checkpoint "$LATEST" \
  --best-checkpoint "$BEST" \
  --export-checkpoint "$POLICY" \
  --log-jsonl "$LOG" \
  --train-opponents "${TRAIN_OPPONENTS[@]}" \
  --eval-opponents "${EVAL_OPPONENTS[@]}" \
  --benchmark-opponents "${BENCHMARK_OPPONENTS[@]}"
