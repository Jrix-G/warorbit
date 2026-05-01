import time
from SimGame import run_match
import bot_v8
from opponents import ZOO

opp_name = "notebook_orbitbotnext"
opp = ZOO[opp_name]
t0 = time.time()
r = run_match([bot_v8.agent, opp], seed=123, n_players=2)
dt = time.time() - t0
winner = int(r.get("winner", -1))
print(f"opp = {opp_name}")
print(f"winner = {winner} ({'V8.1' if winner == 0 else opp_name if winner == 1 else 'tie'})")
print(f"steps = {r.get('steps')}")
print(f"time  = {dt:.1f}s")
