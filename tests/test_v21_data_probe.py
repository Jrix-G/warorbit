import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v21_data_probe
import v21_dataset


def test_probe_npz_reports_policy_stats(tmp_path):
    path = tmp_path / "data.npz"
    pol = np.zeros((2, 3, 4), dtype=np.float32)
    pol[:, :, 0] = 1.0
    mask = np.array([[True, True, False], [True, False, False]])
    np.savez(path, POL=pol, MASK=mask, EP=np.array([1, 2]))
    report = v21_data_probe.probe_npz(path)
    assert report["episodes"] == 2
    assert report["policy"]["pass_rate"] == 1.0


def test_probe_jsonl_reports_sources(tmp_path):
    path = tmp_path / "samples.jsonl"
    sample = {
        "state": {"x": 1},
        "candidates": [{"features": [0.0], "shot": []}],
        "chosen": {"features": [0.0], "shot": []},
        "outcome": 0.0,
        "esc": 0.0,
        "episode_id": "ep",
        "player": 0,
        "n_players": 2,
        "source": "unit",
    }
    v21_dataset.write_jsonl(path, [sample])
    report = v21_data_probe.probe_jsonl(path)
    assert report["samples"] == 1
    assert report["sources"]["unit"] == 1


def test_probe_generic_jsonl_handles_bench_log(tmp_path):
    path = tmp_path / "bench.jsonl"
    path.write_text(
        '{"type":"meta"}\n{"type":"game","mode":"4p","win":1}\n',
        encoding="utf-8",
    )
    report = v21_data_probe.probe_jsonl(path)
    assert report["kind"] == "jsonl-generic"
    assert report["games"] == 1
    assert report["modes"]["4p"] == 1
