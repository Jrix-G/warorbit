import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v21_gate


def test_evaluate_gate_passes_reasonable_summary():
    report = v21_gate.evaluate_gate(
        {
            "games": 24,
            "candidate_errors": 0,
            "modes": {
                "2p": {"total": 12, "winrate": 0.8, "ci_low": 0.5},
                "4p": {"total": 12, "winrate": 0.6, "ci_low": 0.3},
            },
        }
    )
    assert report["passed"]


def test_evaluate_log_reads_v20_bench_jsonl(tmp_path):
    path = tmp_path / "bench.jsonl"
    rows = [
        {"type": "meta"},
        {"type": "game", "mode": "4p", "n_players": 4, "our_seat": 0, "all_scores": [10, 5, 3, 1], "win": 1, "score_margin": 5},
        {"type": "game", "mode": "4p", "n_players": 4, "our_seat": 1, "all_scores": [10, 15, 3, 1], "win": 1, "score_margin": 5},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary = v21_gate.summarize_log(path)
    assert summary["games"] == 2
    assert summary["modes"]["4p"]["wins"] == 2


def test_gate_rejects_negative_4p_margin():
    report = v21_gate.evaluate_gate(
        {
            "games": 24,
            "candidate_errors": 0,
            "modes": {
                "2p": {"total": 12, "winrate": 0.8, "ci_low": 0.5, "avg_margin": 10.0},
                "4p": {"total": 12, "winrate": 0.6, "ci_low": 0.3, "avg_margin": -1.0},
            },
        },
        rules={"min_avg_margin_4p": 0.0},
    )
    assert not report["passed"]
    assert "min_avg_margin_4p" in report["failures"]
