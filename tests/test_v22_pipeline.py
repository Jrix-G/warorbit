import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import v15_search
import v22_agent
import v22_collect
import v22_dataset
import v22_features
import v22_model
import v22_oracle
import v22_promote
import v22_train_combo


def test_combo_features_size():
    fs = v22_oracle._smoke_state()
    shot = v22_oracle._ranked_shots(fs, 0, top_k=1)[0]
    feat = v22_features.combo_features(fs, 0, [shot])
    assert feat.shape[0] == len(v22_features.FEATURE_NAMES)
    assert np.isfinite(feat).all()


def test_oracle_sample_roundtrip(tmp_path):
    sample = v22_oracle.sample_from_state(
        v22_oracle._smoke_state(),
        0,
        episode_id="ep",
        horizon=3,
        det_horizon=2,
        top_k=4,
        beam_width=8,
        min_advantage=-1.0,
    )
    assert sample["combos"]
    assert 0 <= sample["chosen"] < len(sample["combos"])
    path = tmp_path / "v22.jsonl"
    assert v22_dataset.write_jsonl(path, [sample]) == 1
    assert len(v22_dataset.load_jsonl(path)) == 1


def test_train_combo_ranker_smoke():
    sample = v22_oracle.sample_from_state(
        v22_oracle._smoke_state(),
        0,
        episode_id="ep",
        horizon=3,
        det_horizon=2,
        top_k=4,
        beam_width=8,
        min_advantage=-1.0,
    )
    model, metrics = v22_train_combo.train_combo_ranker([sample], epochs=5, lr=0.05)
    assert model.ready_for(len(v22_features.FEATURE_NAMES))
    assert metrics["samples"] == 1.0


def test_model_save_load(tmp_path):
    model = v22_model.LinearComboRanker(w=np.ones(len(v22_features.FEATURE_NAMES)), b=0.25)
    path = tmp_path / "ranker.npz"
    v22_model.save(path, model)
    loaded = v22_model.LinearComboRanker.load(path)
    assert loaded.ready_for(len(v22_features.FEATURE_NAMES))


def test_agent_search_smoke():
    fs = v22_oracle._smoke_state()
    obs = v15_search.state_to_obs(fs, 0)
    move = v22_agent.agent(obs, {"nPlayers": 2, "episodeSteps": 120}, time_budget=0.05, horizon=4)
    assert isinstance(move, list)


def test_collect_smoke():
    cfg = v22_collect.CollectConfig(n_players=2, games=1, steps=5, stride=2, horizon=2, det_horizon=2, top_k=3, beam_width=4, max_samples=2, policy="pass")
    samples = v22_collect.collect(cfg)
    assert len(samples) <= 2


def test_promote_rejects_tiny_dataset(tmp_path):
    sample = v22_oracle.sample_from_state(
        v22_oracle._smoke_state(),
        0,
        episode_id="ep",
        horizon=3,
        det_horizon=2,
        top_k=4,
        beam_width=8,
        min_advantage=-1.0,
    )
    dataset = tmp_path / "v22.jsonl"
    bench = tmp_path / "bench.jsonl"
    v22_dataset.write_jsonl(dataset, [sample])
    bench.write_text(
        '{"type":"meta","workers":1}\n'
        '{"type":"game","mode":"4p","n_players":4,"all_scores":[10,9,8,7],"our_seat":0,"win":1,"candidate_errors":0}\n',
        encoding="utf-8",
    )
    report = v22_promote.evaluate(dataset, bench, rules={"min_samples": 2, "min_games_total": 1, "min_games_4p": 1, "min_wr_4p": 0.0, "min_ci_low_4p": 0.0})
    assert not report["passed"]
    assert "min_samples" in report["failures"]
