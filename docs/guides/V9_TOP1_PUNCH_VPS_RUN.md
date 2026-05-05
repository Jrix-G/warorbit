# V9 top1 punch - lancement VPS

Date: 2026-05-05

Ce run lance V9 depuis zero avec le patch strategique derive des replays top1.
Il ne reprend aucun best precedent dans le namespace choisi, car le script supprime
`latest`, `best`, `policy`, `log` et les snapshots du run avant de demarrer.

## Objectif

Tester si le comportement top1 améliore rapidement le 4p sans casser le 2p:

- captures early plus massives;
- minimum `14` ships en 2p sur les vraies captures opening;
- minimum `16` ships en 4p sur les vraies captures opening;
- minimum `24` ships en 4p midgame sur les vraies captures;
- filtre de distance 4p opening pour eviter les neutres lointains non premium;
- snapshots a chaque generation pour recuperer un pic type gen04.

## Commande VPS

A lancer dans le SSH, depuis `~/warorbit`, apres avoir pull/synchronise le code:

```bash
cd ~/warorbit && \
chmod +x scripts/run_v9_fresh_4p_guardian.sh && \
RUN_NAME=v9_top1_punch_8h ./scripts/run_v9_fresh_4p_guardian.sh 2>&1 | tee evaluations/v9_top1_punch_8h.console.log
```

## Fichiers produits

```text
evaluations/v9_top1_punch_8h_latest.npz
evaluations/v9_top1_punch_8h_best.npz
evaluations/v9_top1_punch_8h_policy.npz
evaluations/v9_top1_punch_8h_train.jsonl
evaluations/v9_top1_punch_8h.console.log
evaluations/v9_top1_punch_8h_snapshots/gen_0001_train.npz
evaluations/v9_top1_punch_8h_snapshots/gen_0001_policy.npz
...
```

## Lecture rapide des logs

Continuer si:

```text
bench4p monte vers 0.42+
fronts descend vers <= 2.70
2p reste fort
sel progresse
```

Surveiller:

```text
block=bench4p_low
block=fronts_high
bench=(2p ... 4p ...)
4pdiag=WARN/OK
xfer=...
bb=...
fronts=...
```

Un run utile peut avoir `promo=0` pendant plusieurs generations. Les snapshots
permettent quand meme de recuperer une generation forte meme si elle n'est pas
promue en `best`.

## Recuperation d'un snapshot

Exemple pour recuperer gen 4:

```bash
ls -lh evaluations/v9_top1_punch_8h_snapshots/
cp evaluations/v9_top1_punch_8h_snapshots/gen_0004_policy.npz evaluations/v9_top1_punch_8h_gen0004_policy.npz
```

