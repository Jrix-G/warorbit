import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v21_collect_oracle
import v21_dataset


def test_collect_oracle_samples_smoke():
    cfg = v21_collect_oracle.CollectConfig(n_players=2, games=1, steps=6, stride=3, horizon=3, top_k=3, max_samples=2)
    samples = v21_collect_oracle.collect_oracle_samples(cfg)
    assert samples
    assert len(samples) <= 2
    assert samples[0]["source"] == "oracle_pass"


def test_write_samples(tmp_path):
    cfg = v21_collect_oracle.CollectConfig(n_players=2, games=1, steps=4, stride=2, horizon=2, top_k=2, max_samples=1)
    samples = v21_collect_oracle.collect_oracle_samples(cfg)
    path = tmp_path / "samples.jsonl"
    assert v21_collect_oracle.write_samples(str(path), samples) == len(samples)
    assert len(v21_dataset.load_jsonl(path)) == len(samples)
