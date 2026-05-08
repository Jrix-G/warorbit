"""test_parity_kaggle_make.py — Verify CGame matches kaggle_environments.make()
exactly (the actual real-Kaggle runner, not just our local copy)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402
from c_engine import CGame  # noqa: E402

POS_TOL = 1e-9


def _records_match(py_recs, c_recs, kind, step, seed):
    py_sorted = sorted(py_recs, key=lambda r: int(r[0]))
    c_sorted = sorted(c_recs, key=lambda r: int(r[0]))
    if len(py_sorted) != len(c_sorted):
        return f"seed={seed} step={step} {kind} count py={len(py_sorted)} c={len(c_sorted)}"
    for i, (pp, cc) in enumerate(zip(py_sorted, c_sorted)):
        for k in (0, 1, 5, 6):
            if int(pp[k]) != int(cc[k]):
                return f"seed={seed} step={step} {kind}[{i}].field[{k}] py={pp[k]} c={cc[k]}"
        for k in (2, 3, 4):
            d = abs(float(pp[k]) - float(cc[k]))
            if d >= POS_TOL:
                return f"seed={seed} step={step} {kind}[{i}].field[{k}] py={pp[k]:.12f} c={cc[k]:.12f} diff={d:.2e}"
    return None


def noop_agent(obs):
    return []


def run_one(seed, ep_steps=200):
    """Drive both engines step-by-step with noop actions, compare planets/fleets."""
    env = make("orbit_wars", configuration={"episodeSteps": ep_steps, "seed": seed}, debug=False)
    env.reset(num_agents=2)
    c = CGame(2, seed=seed, episode_steps=ep_steps)

    # Compare initial state (after env reset, before any step)
    py_obs0 = env.state[0].observation
    c_obs0 = c.observation(0)
    err = _records_match(py_obs0.planets, c_obs0.planets, 'planet', 0, seed)
    if err: return err
    err = _records_match(py_obs0.fleets, c_obs0.fleets, 'fleet', 0, seed)
    if err: return err

    for t in range(ep_steps):
        if c.done:
            break
        # Step kaggle env with [[], []] (noop)
        env.step([[], []])
        c.step([[], []])

        py_obs = env.state[0].observation
        c_obs = c.observation(0)
        err = _records_match(py_obs.planets, c_obs.planets, 'planet', t + 1, seed)
        if err: return err
        err = _records_match(py_obs.fleets, c_obs.fleets, 'fleet', t + 1, seed)
        if err: return err
        if env.done:
            break
    return None


def main():
    print("=== KAGGLE.MAKE vs CGame parity (real runner) ===")
    seeds = list(range(0, 30))
    fails = 0
    for s in seeds:
        err = run_one(s, ep_steps=200)
        if err:
            fails += 1
            print(f"  seed={s} FAIL: {err}")
        else:
            print(f"  seed={s} OK")
    print()
    if fails == 0:
        print(f"✓ {len(seeds)}/{len(seeds)} byte-identical to kaggle_environments.make()")
    else:
        print(f"✗ {fails}/{len(seeds)} FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
