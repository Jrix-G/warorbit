# ===============================================================================
# Orbit Wars — overnight ES training (7h budget)
# ===============================================================================
# Build informé par :
#   - docs/analysis/critique_training_v7.txt (variance, sample size, plafond)
#   - docs/analysis/reverse_engineering_top1.txt (kovi behavioral fingerprint)
#
# Diagnostic du run precedent (gen 75 -> 105) :
#   best=43% (gen 90), final=37% sur les MEMES best_params -> pure variance d'eval
#   (sigma_hat = 0.09 sur 30 games). |p| derive 2.74 -> 3.00 sans gain WR :
#   marche aleatoire car SNR ~ 5e-4 par generation.
#
# Strategie de ce run :
#   1. Reduire la variance par-perturbation : pairs 2->4, games-per-eval 1->2
#      => 16 reward/gen (vs 4 actuel) => SNR x4
#   2. Reduire la variance d'evaluation : eval-games-per-opp 2->3 (45 games)
#      => sigma_hat ~ 0.074 (vs 0.09)
#   3. Monitorer plus souvent : eval-every 15->10 (rollback active plus tot)
#   4. Resserrer le rollback : margin 0.05->0.04 (viable car eval moins bruite)
#   5. sigma-decay-on-rollback 0.7->0.6 : reduire plus vite si on derive
#   6. lr 0.02->0.015 et sigma-init 0.15->0.12 : pas plus prudents
#   7. episode-steps 150->220 : capturer la "conversion threshold" t60-t100
#      identifiee dans le reverse-engineering (planets >= 13 a t=80)
#   8. Auto-resume best (ratchet sur le checkpoint le plus connu)
#   9. Baseline eval ON : on ancre le score reel de depart au demarrage
#
# Budget temps estime :
#   pairs=4 gpe=2 = 16 tasks/gen, workers=7 = 3 batches (with one spare slot)
#   match @ eps=220 ~7s par jeu en parallele
#   gen_t attendu : 60-90s (tolerance 50% si charge variable)
#   eval (45 jeux / 8 workers) ~ 60s toutes les 10 gens
#   => 280-360 generations sur 7h
# ===============================================================================

$env:PYTHONIOENCODING = 'utf-8'

python -u .\train_kaggle.py `
    --minutes 420 `
    --workers 7 `
    --pairs 4 `
    --games-per-eval 2 `
    --eval-games-per-opp 3 `
    --match-4p-ratio 0.7 `
    --eval-match-4p-ratio 0.7 `
    --episode-steps 220 `
    --lr 0.015 `
    --sigma-init 0.12 `
    --sigma-min 0.04 `
    --momentum 0.9 `
    --eval-every 10 `
    --resume-source best `
    --rollback-on-bad-eval `
    --rollback-margin 0.04 `
    --sigma-decay-on-rollback 0.6 `
    --out evaluations\scorer_v7_kaggle `
    2>&1 | Tee-Object -FilePath .\training_kaggle_7h.log
