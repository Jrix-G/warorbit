#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs evaluations

stamp="$(date +%Y%m%d_%H%M%S)"
log_path="logs/v9_guardian_8h_vps_${stamp}.log"

if command -v python3 >/dev/null 2>&1; then
  py="python3"
else
  py="python"
fi

train_opponents=(
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

args=(
  ./run_v9.py
  --minutes 480
  --hard-timeout-minutes 480
  --workers 4
  --pairs 6
  --games-per-eval 3
  --eval-games 10
  --benchmark-games 12
  --min-promotion-benchmark-games 12
  --benchmark-progress-every 4
  --eval-every 1
  --benchmark-every 1
  --max-steps 95
  --eval-max-steps 190
  --four-player-ratio 0.84
  --eval-four-player-ratio 0.84
  --train-search-width 3
  --train-simulation-depth 0
  --train-simulation-rollouts 0
  --train-opponent-samples 1
  --front-lock-turns 20
  --target-active-fronts 2.0
  --target-backbone-turn-frac 0.15
  --front-penalty-weight 0.050
  --front-penalty-cap 0.12
  --front-ok-bonus 0.065
  --front-partial-bonus 0.030
  --backbone-penalty-weight 0.120
  --backbone-bonus-weight 0.095
  --front-pressure-plan-bias 0.15
  --front-pressure-attack-penalty 0.13
  --guardian-enabled 1
  --guardian-min-benchmark-4p 0.40
  --guardian-min-benchmark-backbone 0.07
  --guardian-max-benchmark-fronts 2.80
  --guardian-max-generalization-gap 0.20
  --export-best-on-finish 1
  --min-benchmark-score 0.40
  --max-generalization-gap 0.20
  --exploration-rate 0.075
  --reward-noise 0.008
  --pool-limit 15
  --checkpoint evaluations/v9_guardian_8h_vps_latest.npz
  --best-checkpoint evaluations/v9_guardian_8h_vps_best.npz
  --export-checkpoint evaluations/v9_guardian_8h_vps_policy.npz
  --log-jsonl evaluations/v9_guardian_8h_vps_train.jsonl
  --train-opponents
  "${train_opponents[@]}"
)

if [[ "${1:-}" == "--fresh" || "${1:-}" == "-Fresh" || "${1:-}" == "fresh" ]]; then
  args+=(--no-resume)
fi

echo "[run_v9_guardian_8h_vps] starting | workers=4 | pairs=6 | duration=8h | log=${log_path}"
"${py}" "${args[@]}" 2>&1 | tee "${log_path}"
