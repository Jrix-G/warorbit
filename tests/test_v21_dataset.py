import pytest

from v21_dataset import (
    V21SampleError,
    dedupe_samples,
    mix_offline_online,
    normalize_sample,
    sample_key,
    split_by_episode,
)


def sample(**overrides):
    base = {
        "state": {"turn": 3, "planets": [[0, 1, 12]]},
        "candidates": [{"move": "pass"}, {"move": "attack", "target": 2}],
        "chosen": {"move": "attack", "target": 2},
        "outcome": 1,
        "esc": 0.25,
        "episode_id": "ep-a",
        "player": 0,
        "n_players": 2,
        "source": "offline",
    }
    base.update(overrides)
    return base


def test_normalize_sample_canonicalizes_types():
    out = normalize_sample(sample(outcome="1", esc="0.5", episode_id=42))

    assert out["outcome"] == 1.0
    assert out["esc"] == 0.5
    assert out["episode_id"] == "42"
    assert list(out) == [
        "state",
        "candidates",
        "chosen",
        "outcome",
        "esc",
        "episode_id",
        "player",
        "n_players",
        "source",
    ]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"candidates": []}, "candidates must be non-empty"),
        ({"chosen": {"move": "missing"}}, "chosen must match one candidate"),
        ({"outcome": float("nan")}, "outcome must be finite"),
        ({"player": 2}, "player must satisfy"),
        ({"n_players": 1}, "n_players must be >= 2"),
        ({"source": ""}, "source must be non-empty"),
    ],
)
def test_normalize_sample_rejects_bad_inputs(overrides, message):
    with pytest.raises(V21SampleError, match=message):
        normalize_sample(sample(**overrides))


def test_sample_key_ignores_source_but_not_decision_content():
    first = sample(source="offline")
    duplicate_from_online = sample(source="online")
    different_choice = sample(chosen={"move": "pass"})

    assert sample_key(first) == sample_key(duplicate_from_online)
    assert sample_key(first) != sample_key(different_choice)
    assert len(dedupe_samples([first, duplicate_from_online, different_choice])) == 2


def test_split_by_episode_keeps_episodes_intact():
    samples = [
        sample(episode_id="ep-a", player=0),
        sample(episode_id="ep-a", player=1, chosen={"move": "pass"}),
        sample(episode_id="ep-b", player=0, state={"turn": 9}),
        sample(episode_id="ep-b", player=1, state={"turn": 10}, chosen={"move": "pass"}),
    ]

    train, val = split_by_episode(samples, val_fraction=0.5, seed=7)
    train_ids = {item["episode_id"] for item in train}
    val_ids = {item["episode_id"] for item in val}

    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {"ep-a", "ep-b"}


def test_mix_offline_online_limits_online_fraction_and_dedupes():
    offline = [
        sample(episode_id="off-1", source="offline"),
        sample(episode_id="off-2", state={"turn": 4}, source="offline"),
        sample(episode_id="off-3", state={"turn": 5}, source="offline"),
    ]
    online = [
        sample(episode_id="on-1", state={"turn": 6}, source="online"),
        sample(episode_id="on-2", state={"turn": 7}, source="online"),
        sample(episode_id="off-1", source="online"),
    ]

    mixed = mix_offline_online(offline, online, online_fraction=0.25, seed=1)

    assert len(mixed) == 4
    assert sum(item["source"] == "online" for item in mixed) == 1
    assert len({sample_key(item) for item in mixed}) == len(mixed)
