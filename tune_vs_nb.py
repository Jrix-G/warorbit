"""Tune submission constants vs notebook_physics_accurate."""
import time, math, random, importlib.util, sys, json, os
sys.path.insert(0, '/home/jason/Documents/warorbit')
os.chdir('/home/jason/Documents/warorbit')
spec = importlib.util.spec_from_file_location('sub', 'submission.py')
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
from SimGame import run_match
from opponents import ZOO

OPP = ZOO['notebook_physics_accurate']

def sub_fast(o, c=None):
    if isinstance(o, dict):
        o = dict(o); o['remainingOverageTime'] = 0.5
    return sub.agent(o, c)

PARAMS = ['FLEET_SEND_RATIO', 'THREAT_THRESHOLD', 'EARLY_GAME_DEFENSE',
          'MID_GAME_DEFENSE', 'LATE_GAME_DEFENSE',
          'EARLY_HORIZON', 'MID_HORIZON', 'LATE_HORIZON']
RANGES = {
    'FLEET_SEND_RATIO': (0.55, 0.95),
    'THREAT_THRESHOLD': (0.10, 0.50),
    'EARLY_GAME_DEFENSE': (0.05, 0.40),
    'MID_GAME_DEFENSE': (0.05, 0.35),
    'LATE_GAME_DEFENSE': (0.00, 0.30),
    'EARLY_HORIZON': (20, 90),
    'MID_HORIZON': (60, 180),
    'LATE_HORIZON': (80, 250),
}

def setp(p):
    for k, v in p.items():
        setattr(sub, k, v)

def evalp(p, n=4, seed_base=1000):
    setp(p); w = l = t = 0
    for i in range(n):
        if i % 2 == 0:
            r = run_match([sub_fast, OPP], seed=seed_base + i, max_steps=500)
        else:
            r = run_match([OPP, sub_fast], seed=seed_base + i, max_steps=500)
        wn = r['winner']
        if wn == -1:
            t += 1
        elif (i % 2 == 0 and wn == 0) or (i % 2 == 1 and wn == 1):
            w += 1
        else:
            l += 1
    return w + 0.5 * t, w, l, t

current = {k: getattr(sub, k) for k in PARAMS}
t0 = time.time()
score, w, l, ties = evalp(current, n=10, seed_base=500)
print(f'CURRENT 10g: {w}W {l}L {ties}T  ({time.time()-t0:.1f}s)', flush=True)

best = current; best_score = score
budget = 280
ts = time.time()
trials = 0
while time.time() - ts < budget:
    cand = {}
    for k in PARAMS:
        lo, hi = RANGES[k]
        if isinstance(current[k], int):
            cand[k] = random.randint(int(lo), int(hi))
        else:
            cand[k] = round(random.uniform(lo, hi), 3)
    sc, w, l, ties = evalp(cand, n=4, seed_base=2000 + trials * 11)
    trials += 1
    if sc > best_score:
        best_score = sc
        best = cand
        print(f'  t{trials} {time.time()-ts:.0f}s NEW BEST {w}W{l}L{ties}T -> {cand}', flush=True)

print(f'\nTrials: {trials}, total {time.time()-t0:.1f}s', flush=True)
print('BEST:', best, flush=True)
score, w, l, ties = evalp(best, n=10, seed_base=9999)
print(f'BEST 10g re-eval: {w}W {l}L {ties}T', flush=True)
with open('/tmp/tuned_vs_nb.json', 'w') as f:
    json.dump(best, f)
print('saved /tmp/tuned_vs_nb.json', flush=True)
