# 📺 Guide Extraction Données Replay

## Situation
- Tu regardes: Episode 75514378 (kovi perd malgré 2578 ships vs 737 du gagnant)
- Total: 220 tours
- Joueurs: Mahdieh Rezaie (WINNER 737), kovi (LOSER 2578), yuto083 (914), Arne (538)

## Options pour me donner les données

### Option 1: RAPIDE - Décris juste les événements clés

Copie-colle directement ici les moments importants que tu vois:

```
Turn 1: Starting position
  - Mahdieh: X ships (ou "peu de données visibles")
  - kovi: Y ships
  - [Observations]

Turn 20: [Key event]
  - Event: Mahdieh/kovi attacks Planet X
  - [Observations]

Turn 50: FIRST COMET
  - Comet captured by: [Mahdieh/kovi/other]
  - [Observations]

Turn 100: Mid-game state
  - Mahdieh: X ships
  - kovi: Y ships
  - [Observations]

Turn 150: SECOND COMET
  - Comet captured by: [Mahdieh/kovi/other]
  - [Observations]

Turn 200: Late game
  - Mahdieh: X ships
  - kovi: Y ships
  - [Observations]

Turn 220: Final state
  - Mahdieh WINS with 737 ships
  - kovi LOSES with 2578 ships
```

### Option 2: DÉTAILLÉ - Remplis le JSON template

1. Ouvre: `game_data_template.json`
2. Regarde le replay tour par tour
3. Remplis chaque turn avec les chiffres et observations
4. Sauvegarde et dis-moi que c'est prêt

### Option 3: SCREENSHOTS

Prends des screenshots des moments clés:
- Turn 1 (start)
- Turn 50 (first comet)
- Turn 100 (mid-game)
- Turn 150 (second comet)
- Turn 220 (end)

Pour chaque screenshot, note:
- Tour number
- Ships par joueur (visible dans UI)
- Événement clé

## Données clés que je cherche

### Pour Mahdieh (le gagnant - 737 ships finaux)
1. **Première attaque** (quel turn, quelle planète?)
2. **Ratio d'envoi de flottes** (X% de ses ships/tour?)
3. **Capture des comètes** (lesquelles? turns 50/150/250/350/450?)
4. **Évitement du soleil** (waypoints ou routes directes?)
5. **Agressivité vs défense** (quand passe-t-il de l'un à l'autre?)

### Pour kovi (le perdant - 2578 ships finaux)
1. **Où a-t-il concentré ses attaques?**
2. **Pourquoi n'a-t-il pas écrasé Mahdieh qui avait 3.5x moins de ships?**
3. **A-t-il raté les comètes?**
4. **Moment critique où il a commencé à perdre?**

## Format simplifié si tu veux juste donner les éléments texte

```
Turn X: [Description de ce qui se passe]
Turn Y: [Description de ce qui se passe]
Key moment Z: [Moment décisif]
Mahdieh strategy: [Ce qu'il fait de différent]
kovi mistake: [Où il s'est trompé]
```

## Exemple de ce que j'attends

```
Turn 1: Both players start with initial planet sets
Turn 15: Mahdieh sends fleet to attack neutral Planet 3 (costs 50 ships, captures 40-production planet)
Turn 30: kovi attacks neutral Planet 7 (commits 200 ships, wastes resources)
Turn 50: FIRST COMET - Mahdieh captures with only 30 ships sent (smart timing!)
Turn 100: Mahdieh has 380 ships, kovi has 1800 ships (but spread thin)
Turn 150: SECOND COMET - kovi takes this one, but too late
Turn 220: Mahdieh 737 (concentrated), kovi 2578 (scattered) - Mahdieh WINS because ship consolidation beats raw numbers
```

## Commandes pour analyser une fois que tu m'as donné les données

```bash
# Si tu remplis game_data_template.json:
python3 analyze_kovi_loss.py --analyze game_data_template.json

# Si tu mets les données dans un fichier texte:
python3 analyze_competition.py  # Je peux le modifier pour le traiter
```

---

**Envoie-moi les données dans le format que tu préfères. Je peux m'adapter à n'importe quel format - juste besoin de comprendre quand/quoi/qui pour chaque moment clé.**
