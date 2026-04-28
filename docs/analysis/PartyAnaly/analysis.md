# PartyAnaly - Orbit Wars replay analysis

Dataset: submission `52128366`.

## Executive summary

- Episodes analyzed: 13
- Wins: 5
- Losses: 8
- Overall average duration: 276.7 steps
- Average gap to winner: 43.4

## Defeats analysis

The losses are not caused by random opening failures anymore. The trajectory bug is mostly gone. The real failure mode is economic collapse after the opening.

### What the losses have in common

- Average duration in losses: 278.2 steps
- Average first capture turn: 1.0
- Average maximum planets controlled: 9.0
- Average planets at the end: 1.6
- Average peak total ships: 1025.6
- Average score gap to winner: 70.5

### Pattern observed in losses

1. Early capture still happens fast, usually by turn 1.
2. The bot reaches a midgame plateau and then loses board presence.
3. In several defeats, it drops to zero planets for a meaningful stretch.
4. In the stronger losses, it keeps one or two planets alive, but never converts that into a dominant economy.
5. The bot rarely dies because of a single bad shot now. It dies because it stops scaling fast enough.

### Key loss signature

- In 2-player losses, the bot is much weaker: only 1 win out of 5 total 2-player games.
- In the worst losses, the opponent's economy pulls away while our bot remains stuck on too few planets.
- The main issue is not just target selection. It is target selection plus tempo.

### Concrete consequence

The bot often waits too long to become dangerous, then overcommits too late, and loses the economic race.

## Victory analysis

The wins are more stable than the losses. The bot already knows how to capitalize when the map becomes crowded or when the opponent fails to punish expansion.

### What the wins have in common

- Average duration in wins: 274.2 steps
- Average first capture turn: 1.0
- Average maximum planets controlled: 35.8
- Average planets at the end: 32.8
- Average peak total ships: 6404.6

### Pattern observed in wins

1. The bot captures immediately and keeps at least one stable source of production.
2. It scales into multi-planet control instead of stalling at a single front.
3. In four-player games, it benefits from the map being more chaotic and can snowball through openings.
4. The wins usually end with a strong ship advantage, not a narrow tactical miracle.

### Key win signature

- The bot never truly loses the economy in the winning games.
- It maintains enough board presence to keep producing and sending fleets.
- The win condition is usually board control, not a single surgical strike.

## Episode-level view

### Losses

- `75588065`: 2p loss, fast collapse, high score gap.
- `75588278`: 2p loss, long game but opponent outscales us heavily.
- `75588540`: 4p loss, still competitive midgame but not enough scale.
- `75588804`: 4p loss, long game, moderate gap, opponent survives the transition better.
- `75588964`: 4p loss, this is the replay where the old trajectory bug was visible.
- `75589158`: 4p loss, very close score-wise, but still lost the final race.
- `75589379`: 2p loss, close score-wise, but the bot still ends behind.

### Wins

- `75588368`: 2p win, the rare duel win.
- `75588621`: 4p win, steady control.
- `75588708`: 4p win, clean snowball.
- `75588898`: 4p win, solid economy and ship lead.
- `75589069`: 4p win, best overall control in this batch.

## V7 direction

The next version should not be built around more trajectory fixing. That part is already mostly addressed.

The real V7 work should focus on:

- faster first expansion in 2-player games;
- earlier conversion from neutral planets to production;
- stronger target ranking under time pressure;
- better minimum defense so expansion does not leave the home planet exposed;
- explicit dual-mode behavior: duel opening vs four-player opportunistic play.

