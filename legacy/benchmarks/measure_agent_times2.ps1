param(
    [int]$GamesPerOpp = 1,
    [int]$MaxSteps = 500
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

$py = @"
import time
from SimGame import run_match
from opponents import ZOO
import bot_v6

opponents = $($opponents | ForEach-Object { "'$_'" } | Join-String -Separator ", ")
games_per_opp = $GamesPerOpp
max_steps = $MaxSteps

print(f"SimGame timing | games_per_opp={games_per_opp} | max_steps={max_steps} | opponents={len(opponents)}")
total_start = time.perf_counter()

for opp_name in opponents:
    opp = ZOO[opp_name]
    opp_start = time.perf_counter()
    for i in range(games_per_opp):
        if i % 2 == 0:
            run_match([bot_v6.agent, opp], seed=1000 + i, max_steps=max_steps)
        else:
            run_match([opp, bot_v6.agent], seed=1000 + i, max_steps=max_steps)
    elapsed = time.perf_counter() - opp_start
    print(f"{opp_name:<55} {elapsed:6.1f}s")

total_elapsed = time.perf_counter() - total_start
print(f"{'TOTAL':<55} {total_elapsed:6.1f}s")
"@

python -c $py
