param(
    [int]$GamesPerOpp = 1,
    [int]$MaxSteps = 500,
    [double]$OverageTime = 1.0
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

$oppList = ($opponents | ForEach-Object { "'$_'" }) -join ', '

$py = @"
import time
from SimGame import run_match
from opponents import ZOO
import bot_v7

opponents = [$oppList]
games_per_opp = $GamesPerOpp
max_steps = $MaxSteps
overage_time = $OverageTime

print(f"SimGame timing | games_per_opp={games_per_opp} | max_steps={max_steps} | overage_time={overage_time}s | opponents={len(opponents)}")
total_start = time.perf_counter()

for opp_name in opponents:
    if opp_name not in ZOO:
        print(f"{opp_name:<55} SKIP (not in ZOO)")
        continue
    opp = ZOO[opp_name]
    opp_start = time.perf_counter()
    wins = losses = draws = 0
    for i in range(games_per_opp):
        if i % 2 == 0:
            r = run_match([bot_v7.agent, opp], seed=1000+i, max_steps=max_steps, overage_time=overage_time)
            our = 0
        else:
            r = run_match([opp, bot_v7.agent], seed=1000+i, max_steps=max_steps, overage_time=overage_time)
            our = 1
        w = int(r.get('winner', -1))
        if w == our: wins += 1
        elif w == -1: draws += 1
        else: losses += 1
    elapsed = time.perf_counter() - opp_start
    if games_per_opp > 1:
        wr = wins / games_per_opp * 100
        print(f"{opp_name:<55} {elapsed:6.1f}s  W/L/D={wins}/{losses}/{draws}  WR={wr:.0f}%")
    else:
        result = 'WIN' if wins else ('DRAW' if draws else 'LOSS')
        print(f"{opp_name:<55} {elapsed:6.1f}s  {result}")

total_elapsed = time.perf_counter() - total_start
print(f"{'TOTAL':<55} {total_elapsed:6.1f}s")
"@

python -c $py
