from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.src.diagnostics import load_metrics_jsonl, diagnose_run
from neural_network.src.utils import load_json, save_json, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="neural_network/logs/analysis_30min/metrics.jsonl")
    parser.add_argument("--config", default="neural_network/configs/default_config.json")
    parser.add_argument("--output-config", default="neural_network/configs/autocorrected_config.json")
    parser.add_argument("--output-report", default="neural_network/logs/analysis_30min/autocorrect_report.md")
    args = parser.parse_args()

    cfg = load_json(args.config)
    rows = load_metrics_jsonl(args.metrics)
    diag = diagnose_run(rows)

    updated = dict(cfg)
    updated["policy_mode"] = diag.suggested_fix.get("policy_mode", "baseline_first")
    updated["temperature"] = diag.suggested_fix.get("temperature", 1.0)
    updated["min_ratio"] = diag.suggested_fix.get("min_ratio", 0.1)
    updated["explore"] = diag.suggested_fix.get("explore", False)

    ensure_dir(Path(args.output_config).parent)
    save_json(args.output_config, updated)

    report = "\n".join([
        "# Autocorrection report",
        f"- mode: {diag.mode}",
        f"- severity: {diag.severity}",
        "- reasons:",
        *[f"  - {r}" for r in diag.reasons],
        "",
        "## Applied fix",
        f"- policy_mode: {updated['policy_mode']}",
        f"- temperature: {updated['temperature']}",
        f"- min_ratio: {updated['min_ratio']}",
        f"- explore: {updated['explore']}",
        "",
        f"Output config: `{args.output_config}`",
    ])
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text(report, encoding="utf-8")
    print(json.dumps({"diagnosis": diag.mode, "severity": diag.severity, "output_config": args.output_config, "report": args.output_report}, indent=2))


if __name__ == "__main__":
    main()

