param(
    [double]$Minutes = 4.5,
    [int]$Workers = 1,
    [int]$BatchSize = 4,
    [int]$MaxSteps = 80
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
New-Item -ItemType Directory -Force -Path runs, evaluations | Out-Null

$common = @(
    "-u", "train_v14_finetune_v2.py",
    "--minutes", "$Minutes",
    "--workers", "$Workers",
    "--batch-size", "$BatchSize",
    "--max-steps", "$MaxSteps",
    "--load", "evaluations\scorer_v14.npz",
    "--bc-data", "replay_dataset\v14_bc_top1.npz",
    "--selfplay-pool", "4",
    "--selfplay-every", "2"
)

$experiments = @(
    @{
        Name = "base_mixed"
        Args = @("--out", "evaluations\exp_v14_base_mixed.npz", "--out-critic", "evaluations\exp_v14_base_mixed_critic.npz", "--bc-weight-4p", "0.15", "--bc-weight-2p", "0.40", "--lr", "5e-5", "--lr-min", "1e-5", "--temperature-start", "1.1", "--temperature-end", "0.7", "--modes", "4p", "4p", "4p", "2p")
    },
    @{
        Name = "strict4p_lowbc"
        Args = @("--out", "evaluations\exp_v14_strict4p_lowbc.npz", "--out-critic", "evaluations\exp_v14_strict4p_lowbc_critic.npz", "--bc-weight-4p", "0.05", "--bc-weight-2p", "0.0", "--lr", "1e-4", "--lr-min", "5e-5", "--temperature-start", "1.2", "--temperature-end", "0.9", "--modes", "4p", "4p", "4p", "4p", "--rank-reward-4p", "1.0", "0.0", "-0.5", "-1.0")
    },
    @{
        Name = "strict4p_nobc"
        Args = @("--out", "evaluations\exp_v14_strict4p_nobc.npz", "--out-critic", "evaluations\exp_v14_strict4p_nobc_critic.npz", "--no-bc", "--lr", "2e-4", "--lr-min", "5e-5", "--temperature-start", "1.25", "--temperature-end", "0.95", "--modes", "4p", "4p", "4p", "4p", "--rank-reward-4p", "1.0", "0.0", "-0.5", "-1.0", "--value-coef", "0.10", "--entropy-beta", "0.02", "--grad-clip", "2.0", "--ppo-epochs", "1", "--advantage-scale", "1.5")
    }
)

foreach ($exp in $experiments) {
    $log = "runs\v14_$($exp.Name).log"
    Write-Host "Running $($exp.Name) -> $log"
    & python @common @($exp.Args) 2>&1 | Tee-Object -FilePath $log
}

python scripts\v14_analyze_logs.py runs\v14_base_mixed.log runs\v14_strict4p_lowbc.log runs\v14_strict4p_nobc.log
