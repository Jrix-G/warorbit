param(
    [int]$GamesPerOpp = 1
)

$ErrorActionPreference = "Stop"

$opponents = @(
    'notebook_orbitbotnext',
    'notebook_distance_prioritized',
    'notebook_physics_accurate',
    'notebook_tactical_heuristic',
    'notebook_debugendless_orbit_wars_sun_dodging_baseline',
    'notebook_djenkivanov_orbit_wars_optimized_nearest_planet_sniper',
    'notebook_johnjanson_lb_max_score_1000_agi_is_here',
    'notebook_mdmahfuzsumon_how_my_ai_wins_space_wars',
    'notebook_pascalledesma_orbitbotnext',
    'notebook_pascalledesma_orbitwork_v14',
    'notebook_romantamrazov_orbit_star_wars_lb_max_1224',
    'notebook_sigmaborov_lb_928_7_physics_accurate_planner',
    'notebook_sigmaborov_lb_958_1_orbit_wars_2026_reinforce',
    'notebook_sigmaborov_orbit_wars_2026_starter',
    'notebook_sigmaborov_orbit_wars_2026_tactical_heuristic'
)

$python = "python"
$total = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "Benchmark timing | games_per_opp=$GamesPerOpp | opponents=$($opponents.Count)"

foreach ($opp in $opponents) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $python .\benchmark_vs_notebooks.py --games-per-opp $GamesPerOpp --opponents $opp | Out-Null
    $sw.Stop()
    "{0,-55} {1,6:N1}s" -f $opp, $sw.Elapsed.TotalSeconds
}

$total.Stop()
Write-Host ("TOTAL".PadRight(55) + ("{0,6:N1}s" -f $total.Elapsed.TotalSeconds))
