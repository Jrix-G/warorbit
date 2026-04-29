$model = ".\evaluations\v8_policy.npz"
$state = ".\evaluations\v8_policy_state.pkl"
$args = @(
  "--hours", "8.0",
  "--sample-stride", "8",
  "--rollout-steps", "15",
  "--min-oracle-gap", "0.01",
  "--benchmark-games", "10",
  "--benchmark-seconds", "1800",
  "--save-seconds", "900",
  "--skip-initial-benchmark"
)
if (Test-Path $model) {
  $args += @("--resume", $model)
}
if (Test-Path $state) {
  $args += @("--resume-state", $state)
}
python -u .\train_v8.py @args 2>&1 | Tee-Object -FilePath .\training_night.log
