"""Auto-extracted from a Kaggle notebook."""

from IPython.display import HTML, display

display(HTML(r""" 
<div style="max-width: 1480px; margin: 0 auto; padding: 18px 6px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #243042;">
  <div style="background: linear-gradient(180deg, #f7f9fc 0%, #eef4fb 100%); border: 1px solid #d9e2ef; border-radius: 30px; padding: 28px 30px 24px 30px; box-shadow: 0 16px 36px rgba(36, 48, 66, 0.08); overflow: hidden;">
    <div style="font-size: 40px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 10px;">
      🛰️ Structured System Map
    </div>
    <div style="font-size: 20px; line-height: 1.5; color: #5a6b84; max-width: 1180px; margin-bottom: 10px;">
      v11 combines arrival-time ownership forecasting, reinforce-to-hold defense, rescue-versus-recapture timing, multi-source swarm pressure, and crash-window opportunism inside one structured baseline.
    </div>
    <div style="font-size: 17px; line-height: 1.55; color: #6a7890; max-width: 1180px; margin-bottom: 22px;">
      Ships are spent only after three things agree: one direct shot is legal, the target still looks good at the true arrival turn, and the mission remains valid after earlier launches are written into the future.
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 18px;">
      <div style="background: #eef4ff; border: 2px solid #7aa4ff; border-radius: 24px; padding: 18px 20px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 999px; background: #dbe8ff; color: #233246; font-size: 18px; font-weight: 800; margin-bottom: 12px;">1</div>
        <div style="font-size: 21px; font-weight: 800; color: #233246; margin-bottom: 8px;">🧱 Legal Shot</div>
        <div style="font-size: 15px; line-height: 1.65; color: #607089;">Probe several realistic fleet sizes, reject sun-crossing segments, and keep only one direct launch that the rules can actually execute.</div>
      </div>
      <div style="background: #eefaf6; border: 2px solid #6bc38b; border-radius: 24px; padding: 18px 20px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 999px; background: #d7f1e0; color: #233246; font-size: 18px; font-weight: 800; margin-bottom: 12px;">2</div>
        <div style="font-size: 21px; font-weight: 800; color: #233246; margin-bottom: 8px;">🛡️ Future State</div>
        <div style="font-size: 15px; line-height: 1.65; color: #607089;">Replay arrivals, production, and same-turn combat at that ETA so ownership, garrison, and exact need are forecasted instead of guessed.</div>
      </div>
      <div style="background: #fff6ec; border: 2px solid #f2af52; border-radius: 24px; padding: 18px 20px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 999px; background: #ffe2bc; color: #233246; font-size: 18px; font-weight: 800; margin-bottom: 12px;">3</div>
        <div style="font-size: 21px; font-weight: 800; color: #233246; margin-bottom: 8px;">🧯 Hold Logic</div>
        <div style="font-size: 15px; line-height: 1.65; color: #607089;">Split owned-planet decisions into reinforce-to-hold, rescue, and recapture so defense respects fall timing instead of collapsing into one shortcut.</div>
      </div>
      <div style="background: #fff0f6; border: 2px solid #f08cb2; border-radius: 24px; padding: 18px 20px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 999px; background: #ffd6e6; color: #233246; font-size: 18px; font-weight: 800; margin-bottom: 12px;">4</div>
        <div style="font-size: 21px; font-weight: 800; color: #233246; margin-bottom: 8px;">🚀 Mission Layer</div>
        <div style="font-size: 15px; line-height: 1.65; color: #607089;">Spend ships on the best forecasted conversion: single capture, snipe, compact swarm, hostile swarm, post-crash exploit, or one more clean follow-up.</div>
      </div>
      <div style="background: #f7f3ff; border: 2px solid #b79cf7; border-radius: 24px; padding: 18px 20px;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 999px; background: #e7dcff; color: #233246; font-size: 18px; font-weight: 800; margin-bottom: 12px;">5</div>
        <div style="font-size: 21px; font-weight: 800; color: #233246; margin-bottom: 8px;">🔁 Commit Loop</div>
        <div style="font-size: 15px; line-height: 1.65; color: #607089;">Re-aim final sends, append ETA-aware commitments, refresh live doomed checks, and use leftover ships for salvage or rear-to-front staging.</div>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px;">
      <div style="background: rgba(122, 164, 255, 0.08); border: 1px solid #cddcff; border-radius: 22px; padding: 16px 18px;">
        <div style="font-size: 18px; font-weight: 800; margin-bottom: 8px;">☀️ Direct Means Direct</div>
        <div style="font-size: 15px; line-height: 1.6; color: #5f7088;">Sun-crossing lines are discarded. No waypoint route is invented beyond what the game allows.</div>
      </div>
      <div style="background: rgba(107, 195, 139, 0.08); border: 1px solid #cbe9d7; border-radius: 22px; padding: 16px 18px;">
        <div style="font-size: 18px; font-weight: 800; margin-bottom: 8px;">⚔️ Ownership At ETA</div>
        <div style="font-size: 15px; line-height: 1.6; color: #5f7088;">Same-turn arrivals cancel by owner before the garrison fight, so need and hold logic are always arrival-time questions.</div>
      </div>
      <div style="background: rgba(240, 140, 178, 0.08); border: 1px solid #f3cade; border-radius: 22px; padding: 16px 18px;">
        <div style="font-size: 18px; font-weight: 800; margin-bottom: 8px;">🤝 Partial Sources Matter</div>
        <div style="font-size: 15px; line-height: 1.6; color: #5f7088;">Small contributors stay alive long enough to assemble two-source and three-source swarms at one synchronized arrival window.</div>
      </div>
      <div style="background: rgba(183, 156, 247, 0.08); border: 1px solid #d8c8ff; border-radius: 22px; padding: 16px 18px;">
        <div style="font-size: 18px; font-weight: 800; margin-bottom: 8px;">🧭 Refresh The Future</div>
        <div style="font-size: 15px; line-height: 1.6; color: #5f7088;">Every accepted launch rewrites the future. Later missions, salvage, and rear staging all read that updated commitment-aware state.</div>
      </div>
    </div>
  </div>
</div>
"""))

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def parse_version(text):
    parts = []
    for token in text.split('.'):
        digits = ''.join(ch for ch in token if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


required_version = (1, 28, 0)
needs_upgrade = False

try:
    installed_version = parse_version(version('kaggle-environments'))
    needs_upgrade = installed_version < required_version
except PackageNotFoundError:
    needs_upgrade = True

if needs_upgrade:
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'kaggle-environments>=1.28.0']
    )

import kaggle_environments  # noqa: F401

import math
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

# ============================================================
# Shared Configuration
# ============================================================

BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
TOTAL_STEPS = 500
SIM_HORIZON = 110
ROUTE_SEARCH_HORIZON = 60
HORIZON = SIM_HORIZON
LAUNCH_CLEARANCE = 0.1

EARLY_TURN_LIMIT = 40
OPENING_TURN_LIMIT = 80
LATE_REMAINING_TURNS = 60
VERY_LATE_REMAINING_TURNS = 25

SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 2
INTERCEPT_TOLERANCE = 1

SAFE_OPENING_PROD_THRESHOLD = 4
SAFE_OPENING_TURN_LIMIT = 10
ROTATING_OPENING_MAX_TURNS = 13
ROTATING_OPENING_LOW_PROD = 2
FOUR_PLAYER_ROTATING_REACTION_GAP = 3
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.62
FOUR_PLAYER_ROTATING_TURN_LIMIT = 10

COMET_MAX_CHASE_TURNS = 10

ATTACK_COST_TURN_WEIGHT = 0.55
SNIPE_COST_TURN_WEIGHT = 0.45
INDIRECT_VALUE_SCALE = 0.15
INDIRECT_FRIENDLY_WEIGHT = 0.35
INDIRECT_NEUTRAL_WEIGHT = 0.9
INDIRECT_ENEMY_WEIGHT = 1.25

STATIC_NEUTRAL_VALUE_MULT = 1.4
STATIC_HOSTILE_VALUE_MULT = 1.55
ROTATING_OPENING_VALUE_MULT = 0.9
HOSTILE_TARGET_VALUE_MULT = 1.85
OPENING_HOSTILE_TARGET_VALUE_MULT = 1.45
SAFE_NEUTRAL_VALUE_MULT = 1.2
CONTESTED_NEUTRAL_VALUE_MULT = 0.7
EARLY_NEUTRAL_VALUE_MULT = 1.2
COMET_VALUE_MULT = 0.65
SNIPE_VALUE_MULT = 1.12
SWARM_VALUE_MULT = 1.05
REINFORCE_VALUE_MULT = 1.35
CRASH_EXPLOIT_VALUE_MULT = 1.18
FINISHING_HOSTILE_VALUE_MULT = 1.15
BEHIND_ROTATING_NEUTRAL_VALUE_MULT = 0.92

NEUTRAL_MARGIN_BASE = 2
NEUTRAL_MARGIN_PROD_WEIGHT = 2
NEUTRAL_MARGIN_CAP = 8
HOSTILE_MARGIN_BASE = 3
HOSTILE_MARGIN_PROD_WEIGHT = 2
HOSTILE_MARGIN_CAP = 12
STATIC_TARGET_MARGIN = 4
CONTESTED_TARGET_MARGIN = 5
FOUR_PLAYER_TARGET_MARGIN = 3
LONG_TRAVEL_MARGIN_START = 18
LONG_TRAVEL_MARGIN_DIVISOR = 3
LONG_TRAVEL_MARGIN_CAP = 8
COMET_MARGIN_RELIEF = 6
FINISHING_HOSTILE_SEND_BONUS = 3

STATIC_TARGET_SCORE_MULT = 1.18
EARLY_STATIC_NEUTRAL_SCORE_MULT = 1.25
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT = 0.84
DENSE_STATIC_NEUTRAL_COUNT = 4
DENSE_ROTATING_NEUTRAL_SCORE_MULT = 0.86
SNIPE_SCORE_MULT = 1.12
SWARM_SCORE_MULT = 1.06
CRASH_EXPLOIT_SCORE_MULT = 1.05

FOLLOWUP_MIN_SHIPS = 8
LOW_VALUE_COMET_PRODUCTION = 1
LATE_CAPTURE_BUFFER = 5
VERY_LATE_CAPTURE_BUFFER = 3

DEFENSE_LOOKAHEAD_TURNS = 28
DEFENSE_COST_TURN_WEIGHT = 0.4
DEFENSE_FRONTIER_SCORE_MULT = 1.12
DEFENSE_SEND_MARGIN_BASE = 1
DEFENSE_SEND_MARGIN_PROD_WEIGHT = 1
DEFENSE_SHIP_VALUE = 0.55

REINFORCE_ENABLED = True
REINFORCE_MIN_PRODUCTION = 2
REINFORCE_MAX_TRAVEL_TURNS = 22
REINFORCE_SAFETY_MARGIN = 2
REINFORCE_MAX_SOURCE_FRACTION = 0.75
REINFORCE_MIN_FUTURE_TURNS = 40
REINFORCE_HOLD_LOOKAHEAD = 20
REINFORCE_COST_TURN_WEIGHT = 0.35

RECAPTURE_LOOKAHEAD_TURNS = 10
RECAPTURE_COST_TURN_WEIGHT = 0.52
RECAPTURE_VALUE_MULT = 0.88
RECAPTURE_FRONTIER_MULT = 1.08
RECAPTURE_PRODUCTION_WEIGHT = 0.6
RECAPTURE_IMMEDIATE_WEIGHT = 0.4

REAR_SOURCE_MIN_SHIPS = 16
REAR_DISTANCE_RATIO = 1.25
REAR_STAGE_PROGRESS = 0.78
REAR_SEND_RATIO_TWO_PLAYER = 0.62
REAR_SEND_RATIO_FOUR_PLAYER = 0.7
REAR_SEND_MIN_SHIPS = 10
REAR_MAX_TRAVEL_TURNS = 40

PARTIAL_SOURCE_MIN_SHIPS = 6
MULTI_SOURCE_TOP_K = 5
MULTI_SOURCE_ETA_TOLERANCE = 2
MULTI_SOURCE_PLAN_PENALTY = 0.97
HOSTILE_SWARM_ETA_TOLERANCE = 1
THREE_SOURCE_SWARM_ENABLED = True
THREE_SOURCE_MIN_TARGET_SHIPS = 20
THREE_SOURCE_ETA_TOLERANCE = 1
THREE_SOURCE_PLAN_PENALTY = 0.93

PROACTIVE_DEFENSE_HORIZON = 12
PROACTIVE_DEFENSE_RATIO = 0.18
MULTI_ENEMY_PROACTIVE_HORIZON = 14
MULTI_ENEMY_PROACTIVE_RATIO = 0.22
MULTI_ENEMY_STACK_WINDOW = 3
REACTION_SOURCE_TOP_K_MY = 4
REACTION_SOURCE_TOP_K_ENEMY = 4
PROACTIVE_ENEMY_TOP_K = 3

CRASH_EXPLOIT_ENABLED = True
CRASH_EXPLOIT_MIN_TOTAL_SHIPS = 10
CRASH_EXPLOIT_ETA_WINDOW = 2
CRASH_EXPLOIT_POST_CRASH_DELAY = 1

LATE_IMMEDIATE_SHIP_VALUE = 0.6
WEAK_ENEMY_THRESHOLD = 45
ELIMINATION_BONUS = 18.0

BEHIND_DOMINATION = -0.20
AHEAD_DOMINATION = 0.18
FINISHING_DOMINATION = 0.35
FINISHING_PROD_RATIO = 1.25
AHEAD_ATTACK_MARGIN_BONUS = 0.08
BEHIND_ATTACK_MARGIN_PENALTY = 0.05
FINISHING_ATTACK_MARGIN_BONUS = 0.08

DOOMED_EVAC_TURN_LIMIT = 24
DOOMED_MIN_SHIPS = 8

SOFT_ACT_DEADLINE = 0.82
HEAVY_PHASE_MIN_TIME = 0.16
OPTIONAL_PHASE_MIN_TIME = 0.08
HEAVY_ROUTE_PLANET_LIMIT = 32


# ============================================================
# Shared Types
# ============================================================

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


@dataclass(frozen=True)
class ShotOption:
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"
    anchor_turn: int | None = None


@dataclass
class Mission:
    kind: str
    score: float
    target_id: int
    turns: int
    options: list[ShotOption] = field(default_factory=list)
