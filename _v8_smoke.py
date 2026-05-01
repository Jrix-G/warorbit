import time
from SimGame import run_match
import bot_v8
from opponents import ZOO

for opp_name in ['passive', 'random', 'starter']:
    opp = ZOO[opp_name]
    wins = 0
    t0 = time.time()
    n = 4
    for i in range(n):
        agents = [bot_v8.agent, opp] if i % 2 == 0 else [opp, bot_v8.agent]
        r = run_match(agents, seed=42 + i)
        our = 0 if i % 2 == 0 else 1
        if int(r.get('winner', -1)) == our:
            wins += 1
    print(f'{opp_name}: {wins}/{n} in {time.time()-t0:.1f}s')
