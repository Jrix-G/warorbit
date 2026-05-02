from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import pytest

from neural_network.src import notebook_4p_training as n4p
from neural_network.src.autocorrect import apply_autocorrect
from neural_network.src.health_check import analyze_log
from neural_network.src.notebook_4p_training import run_notebook_4p_training
from neural_network.src.utils import load_json


@pytest.fixture
def workspace_tmp():
    root = Path(__file__).resolve().parents[1] / ".test_tmp" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _timed(name, fn):
    started = time.perf_counter()
    try:
        return fn()
    finally:
        print(f"{name}: {time.perf_counter() - started:.3f}s")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _base_config(tmp_path: Path) -> dict:
    cfg = load_json(Path(__file__).resolve().parents[1] / "configs" / "default_config.json")
    cfg.update(
        {
            "train_steps": 10,
            "eval_every": 5,
            "eval_episodes": 20,
            "benchmark_games": 20,
            "max_turns": 8,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "candidate_checkpoint": str(tmp_path / "checkpoints" / "candidate.npz"),
            "best_checkpoint": str(tmp_path / "checkpoints" / "best.npz"),
            "latest_checkpoint": str(tmp_path / "checkpoints" / "latest.npz"),
            "export_path": str(tmp_path / "checkpoints" / "export.npz"),
            "resume_checkpoint": str(tmp_path / "checkpoints" / "best.npz"),
            "notebook_pool_limit": 3,
            "notebook_pool_limit_max": 3,
            "train_notebook_opponents": 3,
        }
    )
    return cfg


def _fake_obs(player: int = 0):
    planets = [
        [0, 0, 0.0, 0.0, 1.0, 50.0, 3.0],
        [1, -1, 10.0, 0.0, 1.0, 15.0, 2.0],
        [2, 1, 20.0, 0.0, 1.0, 30.0, 2.0],
        [3, 2, 30.0, 0.0, 1.0, 30.0, 2.0],
        [4, 3, 40.0, 0.0, 1.0, 30.0, 2.0],
    ]
    return {"player": player, "step": 1, "planets": planets, "fleets": [], "initial_planets": planets}


def _patch_fast_match(monkeypatch):
    def fake_pool(limit=15):
        return ["op_a", "op_b", "op_c"][:limit]

    def opponent(_obs, _config=None):
        return []

    def fake_run_match(agents, seed, n_players=4, max_steps=100):
        for idx, agent in enumerate(agents):
            agent(_fake_obs(idx), None)
        winner = seed % 4
        scores = [float((seed + idx) % 7) for idx in range(4)]
        scores[winner] += 10.0
        return {"winner": winner, "scores": scores, "steps": 4, "final_state": {"my_id": 0, "planets": []}}

    monkeypatch.setattr(n4p, "training_pool", fake_pool)
    monkeypatch.setattr(n4p, "ZOO", {"op_a": opponent, "op_b": opponent, "op_c": opponent, "random": opponent})
    monkeypatch.setattr(n4p, "run_match", fake_run_match)


def test_health_check_detects_healthy_log(workspace_tmp):
    def run():
        log = workspace_tmp / "healthy.jsonl"
        rows = [
            {"policy_entropy": 2.0, "rank": 2, "winner": 0 if i % 3 == 0 else 1, "our_index": 0, "do_nothing_rate": 0.1}
            for i in range(20)
        ]
        _write_jsonl(log, rows)
        report = analyze_log(log, 20)
        assert report["status"] == "healthy"

    _timed("test_health_check_detects_healthy_log", run)


def test_health_check_detects_entropy_collapse(workspace_tmp):
    def run():
        log = workspace_tmp / "collapse.jsonl"
        rows = [
            {"policy_entropy": 0.05, "rank": 2, "winner": 0, "our_index": 0, "do_nothing_rate": 0.1}
            for _ in range(20)
        ]
        _write_jsonl(log, rows)
        report = analyze_log(log, 20)
        assert report["status"] == "unhealthy"
        assert "entropy" in str(report["reason"])

    _timed("test_health_check_detects_entropy_collapse", run)


def test_autocorrect_fixes_entropy_collapse():
    def run():
        cfg = load_json(Path(__file__).resolve().parents[1] / "configs" / "default_config.json")
        health = {"entropy_mean": 0.05, "rank_mean": 2.0, "do_nothing_rate": 0.1, "winrate": 0.25, "window": 20}
        corrected, changes = apply_autocorrect(cfg, health)
        assert changes
        assert corrected["entropy_coef_start"] > cfg["entropy_coef_start"]

    _timed("test_autocorrect_fixes_entropy_collapse", run)


def test_mini_run_then_health_check(workspace_tmp, monkeypatch):
    def run():
        _patch_fast_match(monkeypatch)
        cfg = _base_config(workspace_tmp)
        run_notebook_4p_training(cfg, resume=False)
        log = Path(cfg["log_dir"]) / "notebook_4p_training.jsonl"
        rows = log.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 10
        report = analyze_log(log, 20)
        assert report["status"] in {"healthy", "unhealthy"}

    _timed("test_mini_run_then_health_check", run)


def test_mini_run_autocorrect_cycle(workspace_tmp, monkeypatch):
    def run():
        _patch_fast_match(monkeypatch)
        cfg = _base_config(workspace_tmp / "first")
        run_notebook_4p_training(cfg, resume=False)

        fake_log = workspace_tmp / "unhealthy.jsonl"
        _write_jsonl(fake_log, [{"policy_entropy": 0.05, "rank": 2, "winner": 0, "our_index": 0, "do_nothing_rate": 0.1} for _ in range(20)])
        report = analyze_log(fake_log, 20)
        corrected, changes = apply_autocorrect(cfg, report)
        assert changes
        assert corrected["entropy_coef_start"] != cfg["entropy_coef_start"]

        cfg2 = dict(corrected)
        cfg2["checkpoint_dir"] = str(workspace_tmp / "second" / "checkpoints")
        cfg2["log_dir"] = str(workspace_tmp / "second" / "logs")
        cfg2["candidate_checkpoint"] = str(workspace_tmp / "second" / "checkpoints" / "candidate.npz")
        cfg2["best_checkpoint"] = str(workspace_tmp / "second" / "checkpoints" / "best.npz")
        cfg2["latest_checkpoint"] = str(workspace_tmp / "second" / "checkpoints" / "latest.npz")
        cfg2["export_path"] = str(workspace_tmp / "second" / "checkpoints" / "export.npz")
        run_notebook_4p_training(cfg2, resume=False)
        log = Path(cfg2["log_dir"]) / "notebook_4p_training.jsonl"
        assert len(log.read_text(encoding="utf-8").splitlines()) == 10

    _timed("test_mini_run_autocorrect_cycle", run)
