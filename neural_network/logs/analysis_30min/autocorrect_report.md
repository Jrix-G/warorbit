# Autocorrection report
- mode: terminal_collapse
- severity: high
- reasons:
  - Défaites quasi systématiques sur le vrai environnement
  - avg_reward=-1.000
  - winrate=0.000
  - invalid_rate=0.000
  - filtered_rate=0.000
  - ships_sent=0.00

## Applied fix
- policy_mode: baseline_first
- temperature: 1.4
- min_ratio: 0.1
- explore: True

Output config: `neural_network/configs/autocorrected_config.json`