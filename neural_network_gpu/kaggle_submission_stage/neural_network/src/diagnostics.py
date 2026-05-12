from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json
from pathlib import Path
import numpy as np


@dataclass
class FailureDiagnosis:
    mode: str
    severity: str
    reasons: List[str]
    suggested_fix: Dict[str, Any]


def load_metrics_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def diagnose_run(rows: List[Dict[str, Any]]) -> FailureDiagnosis:
    if not rows:
        return FailureDiagnosis(
            mode="no_data",
            severity="high",
            reasons=["Aucune métrique collectée"],
            suggested_fix={"policy_mode": "baseline", "temperature": 1.0},
        )

    avg_reward = float(np.mean([r.get("avg_reward", 0.0) for r in rows]))
    invalid_rate = float(np.mean([r.get("invalid_actions", 0.0) for r in rows]) / max(1.0, np.mean([r.get("actions", 1.0) for r in rows])))
    filtered_rate = float(np.mean([r.get("avg_filtered_actions", 0.0) for r in rows]))
    winrate = float(np.mean([r.get("final_win", 0.0) for r in rows]))
    ships_sent = float(np.mean([r.get("avg_ships_sent", 0.0) for r in rows]))

    reasons: List[str] = []
    suggested: Dict[str, Any] = {}

    if avg_reward <= -0.9 and winrate <= 0.05:
        reasons.append("Défaites quasi systématiques sur le vrai environnement")
        suggested["policy_mode"] = "baseline_first"
        suggested["temperature"] = 1.4
        suggested["explore"] = True
        mode = "terminal_collapse"
        severity = "high"
    elif invalid_rate > 0.25:
        reasons.append("Trop d'actions invalides")
        suggested["temperature"] = 0.8
        suggested["min_ratio"] = 0.2
        suggested["policy_mode"] = "safe"
        mode = "invalid_action_collapse"
        severity = "high"
    elif ships_sent < 1.0 and filtered_rate > 10.0:
        reasons.append("Politique trop conservatrice ou bloquée")
        suggested["temperature"] = 1.2
        suggested["policy_mode"] = "exploratory"
        suggested["min_ratio"] = 0.05
        mode = "underactive_policy"
        severity = "medium"
    else:
        reasons.append("Signal stable mais pas de progression claire")
        suggested["policy_mode"] = "baseline_first"
        suggested["temperature"] = 1.0
        mode = "stable"
        severity = "low"

    reasons.extend([
        f"avg_reward={avg_reward:.3f}",
        f"winrate={winrate:.3f}",
        f"invalid_rate={invalid_rate:.3f}",
        f"filtered_rate={filtered_rate:.3f}",
        f"ships_sent={ships_sent:.2f}",
    ])
    return FailureDiagnosis(mode=mode, severity=severity, reasons=reasons, suggested_fix=suggested)


def should_promote_checkpoint(eval_scores: List[float], current_best: float, min_episodes: int = 20) -> bool:
    if len(eval_scores) < min_episodes:
        return False
    mean_score = float(np.mean(eval_scores)) if eval_scores else float("-inf")
    return mean_score > float(current_best)
