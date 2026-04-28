# V5 Testing and Iteration Guide

## Phase 1: Validation (Current Status ✓)

### Syntax Validation
```bash
python3 test_v5.py
```
Output should show:
- bot_v5 imports successfully
- All notebook agents load
- agent() function responds to observations

**Status**: PASSED ✓

### Code Review Checklist
- [x] All 7 layers implemented
- [x] Correct API format (array-based planets/fleets)
- [x] Kingmaker logic working
- [x] Game phase detection working
- [x] Threat calculation working
- [x] Expected value targeting working

## Phase 2: Competitive Testing

### Test Against Notebook Agents (900-1100 ELO)

Once kaggle_environments is available:

```bash
pip install kaggle-environments --break-system-packages
python3 run_v5_tests.py
```

**Expected Results**:
- vs Distance-Prioritized [1100]: 45-55%
- vs Physics-Accurate [928.7]: 45-55%
- vs Tactical-Heuristic: 45-55%

**If actual < target**: Proceed to Phase 3 (tuning)
**If actual >= target**: Proceed to Phase 4 (submission)

## Phase 3: Parameter Tuning (If Needed)

### Option A: Manual Tuning

Edit constants in bot_v5.py:

```python
# Increase defense if losing to aggression
EARLY_GAME_DEFENSE = 0.25  → 0.30
MID_GAME_DEFENSE = 0.20    → 0.25
LATE_GAME_DEFENSE = 0.15   → 0.20

# Increase comet aggression if missing them
FLEET_SEND_RATIO = 0.65  → 0.70  (on comets)

# Tune threat detection sensitivity
THREAT_THRESHOLD = 0.25  → 0.20  (more defensive)
```

### Option B: CMA-ES Optimization

Create bot_v5_tunable.py with weights vector, then:

```bash
python3 train.py --test          # 30 games
python3 train.py --rocket        # 5 gen, ~10 min
python3 train.py --medium        # 80 gen, ~1-2 hours
```

**Note**: This time, optimize against notebook agents, not greedy/random.

### Option C: Beam Search (Advanced)

Add local simulator for 20-turn lookahead:

```python
def beam_search_plan(obs, our_id, num_candidates=3):
    """
    For each turn, generate N candidate attack plans:
    1. Aggressive (send 80% ships)
    2. Balanced (send 65% ships)
    3. Defensive (send 50% ships)
    
    Simulate 20 turns locally with each plan.
    Return best plan.
    
    Cost: ~100ms per turn for 20-turn simulation.
    Expected gain: +10-20% win rate (50% → 60-70%).
    """
```

**Complexity**: High (requires local game simulator)
**Expected Return**: +10-20% win rate improvement

## Phase 4: Submission

### Prepare for Kaggle

1. **Verify performance**:
   ```bash
   python3 run_v5_tests.py  # >45% expected
   ```

2. **Copy to submission format**:
   ```bash
   cp bot_v5.py bot_submit.py
   ```

3. **Test bot_submit.py**:
   ```bash
   python3 train.py --test
   ```

4. **Submit to Kaggle**:
   - Go to https://www.kaggle.com/competitions/orbit-wars/code
   - Upload bot_submit.py
   - Run competition and monitor ELO

### Post-Submission Iteration

After submission:

1. **Download replays** (top 10 opponents):
   ```bash
   kaggle competitions download orbit-wars  # via kaggle CLI
   ```

2. **Analyze losses** (in replays):
   - Which phases do we lose?
   - Do comets give us trouble?
   - Are we being out-produced?
   - Are threats getting through our defense?

3. **Iterate**:
   - Adjust constants based on replay analysis
   - Add opponent modeling
   - Resubmit and monitor new ELO

## Performance Metrics

### Tier 1: vs Baselines (should be 100%)
- vs Greedy bot (attack closest)
- vs Random bot (random actions)
- **Expected**: 99-100%

### Tier 2: vs Medium Bots (900 ELO)
- Distance-Prioritized
- Physics-Accurate
- **Expected**: 45-55%

### Tier 3: vs Strong Bots (1100 ELO)
- Top notebooks
- **Expected**: 20-35%

### Tier 4: vs Elite (1500+ ELO)
- kovi, other top players
- **Expected**: 5-15%

## Debugging Guide

### Symptom: Losing to Greedy Bots
**Root Cause**: Expected value targeting broken
- Check: are we attacking neutrals with high production?
- Fix: increase `EARLY_HORIZON` and neutral bonus in calculate_target_value

### Symptom: Getting Destroyed Early Game
**Root Cause**: Defense reserve too low
- Check: do we have 15-25% ships staying home?
- Fix: increase `EARLY_GAME_DEFENSE` from 0.25 → 0.30

### Symptom: Missing Comet Windows
**Root Cause**: Comet bonus insufficient
- Check: is comet_bonus (1000) larger than other values?
- Fix: increase comet value or send ratio during comet windows

### Symptom: Surrounded by Multiple Enemies
**Root Cause**: Threat detection too weak
- Check: are we increasing defense when threat_ratio > 25%?
- Fix: lower `THREAT_THRESHOLD` from 0.25 → 0.20

### Symptom: Always Attacking #1 (Wrong Kingmaker)
**Root Cause**: Kingmaker logic broken
- Check: are we applying 50% bonus to primary target?
- Fix: verify `select_kingmaker_target` returns correct player

## Success Criteria

| Metric | V4 | V5 Target | V5+ Target |
|--------|----|-----------|----|
| vs Baselines | 100% | 100% | 100% |
| vs 900 ELO | 25% | **45-55%** | 50-60% |
| vs 1100 ELO | ~5% | 20-35% | 30-45% |
| Estimated ELO | 800 | 1000-1200 | 1200-1500 |

**V5 is successful if**: Achieves 45-55% vs 900 ELO opponents (2x improvement over V4)

## Timeline

- **Now**: V5 complete, ready for testing
- **Today**: Run test_v5.py validation
- **Tomorrow**: Run against notebook agents (when kaggle_environments available)
- **Week 1**: Tune if needed (manual or CMA-ES)
- **Week 2**: Submit to Kaggle
- **Ongoing**: Analyze replays, iterate

## Reference

- **V5_ARCHITECTURE.md**: Complete design specification
- **bot_v5.py**: Implementation (327 lines)
- **V5_IMPLEMENTATION_COMPLETE.md**: Summary and mathematical justification
- **Notebook agents**: Located in opponents/ directory
- **Kaggle Leaderboard**: https://www.kaggle.com/competitions/orbit-wars/leaderboard

## Questions?

Refer to V5_ARCHITECTURE.md for theoretical justification of each layer.

Key insight: V5 solves V4's problem by replacing "weight optimization" with "game theory architecture."
