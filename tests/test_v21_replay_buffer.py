import pytest

from v21_dataset import V21SampleError, sample_key
from v21_replay_buffer import V21ReplayBuffer


def sample(**overrides):
    base = {
        "state": {"turn": 1},
        "candidates": [{"move": "pass"}, {"move": "attack", "target": 1}],
        "chosen": {"move": "pass"},
        "outcome": -1.0,
        "esc": 0.0,
        "episode_id": "ep-buffer",
        "player": 0,
        "n_players": 2,
        "source": "online",
    }
    base.update(overrides)
    return base


def test_buffer_appends_only_unique_canonical_samples(tmp_path):
    path = tmp_path / "buffer.jsonl"
    buffer = V21ReplayBuffer(path)

    assert buffer.extend([sample(), sample(source="offline")]) == 1
    assert buffer.append(sample(episode_id="ep-2", state={"turn": 2})) is True
    assert buffer.append(sample(episode_id="ep-2", state={"turn": 2})) is False

    rows = path.read_text(encoding="utf-8").strip().splitlines()
    loaded = buffer.load()

    assert len(rows) == 2
    assert len(buffer) == 2
    assert len({sample_key(item) for item in loaded}) == 2
    assert loaded[0]["source"] == "online"


def test_buffer_rejects_invalid_sample_without_appending(tmp_path):
    path = tmp_path / "buffer.jsonl"
    buffer = V21ReplayBuffer(path)

    with pytest.raises(V21SampleError, match="chosen must match one candidate"):
        buffer.extend([sample(episode_id="valid-first"), sample(chosen={"move": "bad"})])

    assert not path.exists()


def test_buffer_sampling_is_deterministic_for_seed(tmp_path):
    buffer = V21ReplayBuffer(tmp_path / "buffer.jsonl")
    buffer.extend(
        sample(episode_id=f"ep-{idx}", state={"turn": idx})
        for idx in range(5)
    )

    first = buffer.sample(3, seed=11)
    second = buffer.sample(3, seed=11)

    assert [item["episode_id"] for item in first] == [
        item["episode_id"] for item in second
    ]
    assert len(first) == 3


def test_buffer_mixes_with_offline(tmp_path):
    buffer = V21ReplayBuffer(tmp_path / "buffer.jsonl")
    buffer.extend(
        [
            sample(episode_id="online-1", state={"turn": 10}, source="online"),
            sample(episode_id="online-2", state={"turn": 11}, source="online"),
        ]
    )
    offline = [
        sample(episode_id="offline-1", state={"turn": 1}, source="offline"),
        sample(episode_id="offline-2", state={"turn": 2}, source="offline"),
        sample(episode_id="offline-3", state={"turn": 3}, source="offline"),
    ]

    mixed = buffer.mix_with_offline(offline, online_fraction=0.25, seed=3)

    assert len(mixed) == 4
    assert sum(item["source"] == "online" for item in mixed) == 1
