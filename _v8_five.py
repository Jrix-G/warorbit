import time
from SimGame import run_match
import bot_v8
from opponents import ZOO

opp_name = "notebook_orbitbotnext"
opp = ZOO[opp_name]

wins = 0
n = 20
t_total = time.time()
for i in range(n):
    if i % 2 == 0:
        agents = [bot_v8.agent, opp]
        our = 0
    else:
        agents = [opp, bot_v8.agent]
        our = 1
    seed = 1000 + i
    t0 = time.time()
    r = run_match(agents, seed=seed, n_players=2)
    dt = time.time() - t0
    winner = int(r.get("winner", -1))
    won = winner == our
    if won:
        wins += 1
    tag = "WIN " if won else "LOSS" if winner != -1 else "TIE "
    print(f"  game {i+1}/{n} seed={seed} our_slot={our} -> {tag} (winner={winner}, steps={r.get('steps')}, {dt:.1f}s)")

print(f"\nV8.1 vs {opp_name}: {wins}/{n} = {100*wins/n:.0f}%  (total {time.time()-t_total:.1f}s)")
