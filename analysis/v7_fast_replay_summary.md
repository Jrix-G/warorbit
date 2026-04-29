# V7 Fast Replay Summary

Source: `replay_dataset/compact/episodes.jsonl.gz`

Status:
- Parsed episodes: `1109`
- File is truncated at the end, but the first ~95% is usable

## Win-rate snapshot

Top submissions by appearance with at least 10 games were not close to 90%.
The best observed win-rates in this slice were:

- `kwon yong deuk` - `48.3%` over `29` games
- `:)` - `40.0%` over `20` games
- `Voikin Ruslan` - `35.4%` over `246` games
- `BaiYuBY` - `35.3%` over `17` games
- `Boruiyang21` - `33.7%` over `181` games

## Behavioral signals

The clearest signal from the replay slice is tempo:

- Winners act more often per turn than losers.
- Winners send more ships per move than losers.
- Winners tend to attack a bit earlier.

Measured on 2-player games:

- Winner first attack median: `4` turns
- Loser first attack median: `4` turns
- Winner actions per turn median: `1.76`
- Loser actions per turn median: `0.70`
- Winner ships sent per move median: `10`
- Loser ships sent per move median: `5`

Game length in 2-player games:

- Median: `175` steps
- Mean: `205.3` steps
- 90th percentile: `332.2` steps

## 4P slice

The dataset also contains a large 4-player slice:

- Parsed 4p episodes: `679`

4p games are materially longer and slightly more conservative:

- Median length: `228` steps
- Mean length: `274.5` steps
- 90th percentile: `500` steps

4p action tempo:

- Winner first attack median: `4` turns
- Loser first attack median: `4` turns
- Winner actions per turn median: `1.79`
- Loser actions per turn median: `0.38`
- Winner ships sent per move median: `11`
- Loser ships sent per move median: `8`

Top 4p win-rate snapshot in this slice:

- `:)` - `40.0%` over `10` games
- `BaiYuBY` - `38.5%` over `13` games
- `Jason Mourier` - `36.4%` over `22` games
- `Voikin Ruslan` - `27.6%` over `156` games
- `Boruiyang21` - `25.4%` over `122` games
- `Reinforced Curiosity` - `20.6%` over `189` games

## Implication for the mix

Yes, the target should treat 4p as the majority regime if that is the intended benchmark mix.
For a `70%` 4p / `30%` 2p training mix, the 4p statistics should drive the default tuning, and the 2p slice should act as a consistency check rather than the primary objective.

## Interpretation for `train_v7_fast.py`

The replay slice supports an aggressive but not reckless profile:

- Keep hostile margins centered near the current V7 defaults.
- Keep `TWO_PLAYER_HOSTILE_AGGRESSION_BOOST` high enough to favor repeated pressure.
- Do not over-increase defensive neutral margins in 2-player play.
- Prefer small search shifts around the current constants rather than widening the ES space.

Practical ranges that look sane from this slice:

- `HOSTILE_MARGIN_BASE`: around `3`
- `HOSTILE_MARGIN_CAP`: around `12`
- `HOSTILE_MARGIN_PROD_WEIGHT`: around `2`
- `NEUTRAL_MARGIN_BASE`: around `2`
- `TWO_PLAYER_HOSTILE_AGGRESSION_BOOST`: around `1.5`
- `ATTACK_COST_TURN_WEIGHT`: around `0.5` to `0.6`
- `STATIC_TARGET_MARGIN`: around `4`

## Notes

- `train_v7_fast.py` already contains the fixes that were identified during the review:
  - jitter before rank shaping
  - per-generation opponent shuffling
  - momentum reset after norm clipping
  - worker-level exception guard

- The remaining work is mostly empirical tuning and validation, not structural repair.
