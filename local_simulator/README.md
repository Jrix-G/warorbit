# Local Orbit Wars Simulator

Simulateur desktop local pour jouer en 2 joueurs contre le `submission.py` actuel, qui embarque les poids V9 best.

## Lancer

Depuis la racine du repo:

```bash
python3 local_simulator/app.py
```

Options utiles:

```bash
python3 local_simulator/app.py --seed 42 --neutral-pairs 8
```

## Controles

- Clic gauche sur une de tes planetes bleues: selection de la source.
- Clic gauche sur une autre planete: ajoute une action en attente.
- Espace: valide le tour, avec tes actions et les actions automatiques du bot V9.
- Clic droit ou Backspace: annule la derniere action en attente.
- `+` / `-`: ajuste le pourcentage de vaisseaux envoyes.
- `B`: affiche ou cache la preview des actions du bot.
- `N`: nouvelle carte.
- `R`: reset de la carte courante.
- `Esc` / `Q`: quitter.

## Notes

- Le moteur utilise `sim.py`, le simulateur local deja aligne avec Kaggle.
- Le bot utilise directement `submission.py`, donc la meme version V9 best que celle preparee pour l'upload Kaggle.
- Les actions sont simultanees: le bot et toi jouez a partir du meme etat, puis le tour avance.

## Tests

Depuis ce dossier:

```bash
cd local_simulator
python3 -m unittest -v
```

