$model = ".\evaluations\v8_policy.npz"
$state = ".\evaluations\v8_policy_state.pkl"
$args = @(
  "--hours", "0.33",
  "--sample-stride", "8",
  "--rollout-steps", "15",
  "--min-oracle-gap", "0.01",
  "--benchmark-games", "10",
  "--benchmark-seconds", "99999",
  "--save-seconds", "60",
  "--skip-initial-benchmark"
)
if (Test-Path $model) {
  $args += @("--resume", $model)
}
if (Test-Path $state) {
  $args += @("--resume-state", $state)
}
python -u .\train_v8.py @args 2>&1 | Tee-Object -FilePath .\training_continue.log
