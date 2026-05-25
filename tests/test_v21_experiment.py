import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v21_experiment


def test_build_record_without_logs():
    record = v21_experiment.build_record("x", artifacts={"a": "b"})
    assert record["name"] == "x"
    assert record["promotable"] is False


def test_write_record(tmp_path):
    path = tmp_path / "record.json"
    v21_experiment.write_record(path, {"name": "x"})
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "x"
