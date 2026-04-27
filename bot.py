"""Orbit Wars — Agent V2.

Améliorations vs V1:
- Poids tunables (vecteur W injecté via WEIGHTS global ou CMA-ES)
- Coordination globale des flottes (pool ships, assignation optimale)
- Stratégie 4P (pénalise attaque du leader, favorise 2e/3e)
- Sun-dodging précis via waypoint intermédiaire
- Prédiction orbite continue (pas entière)
- Comet awareness (bonus si comet faible accessible)
- Détection fin de partie (défense si on domine)
- Threat ETA précise (distance / vitesse flotte ennemie)
"""

import math
from collections import defaultdict

# ── Configuration physique (ne pas changer) ─────────────────────────────────
BOARD_SIZE   = 100.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS   = 10.0
MAX_SPEED    = 6.0
ROT_LIMIT    = 50.0   # orbital_radius + planet_radius < 50 → planète orbite

# ── Poids tunables par CMA-ES ────────────────────────────────────────────────
# Index : W[0..13]
# Pour soumettre le bot avec poids par défaut, laisser WEIGHTS = None.
# train.py injectera les meilleurs poids trouvés dans W_BEST.
WEIGHTS = None  # sera remplacé par CMA-ES

DEFAULT_W = [
    2.0,   # W[0]  neutral_priority   : bonus multiplicatif planètes neutres (✅ consensus)
    12.0,  # W[1]  comet_bonus        : bonus multiplicatif comètes (⬆️ 1.5→12.0, match top players)
    1.0,   # W[2]  production_horizon : nb tours de production estimés dans gain (⬇️ 40.0→1.0, short-term)
    1.25,  # W[3]  distance_penalty   : coût par unité de distance (⬆️ 0.3→1.25, prioritize distance)
    20.0,  # W[4]  defense_reserve    : fraction ships gardée si menacé (⬆️ 0.15→20.0, strong defense)
    1.3,   # W[5]  attack_ratio       : ships_needed × ce ratio avant d'attaquer
    0.6,   # W[6]  fleet_send_ratio   : fraction ships envoyée (après réserve)
    0.5,   # W[7]  leader_penalty     : pénalité si cible est le joueur dominant
    0.4,   # W[8]  weak_enemy_bonus   : bonus si cible <30% de nos ships
    0.05,  # W[9]  sun_waypoint_dist  : distance waypoint soleil (× SUN_RADIUS)
    0.8,   # W[10] endgame_threshold  : ratio ships pour passer en mode défense
    0.25,  # W[11] threat_eta_factor  : poids ETA dans calcul menace
    1.2,   # W[12] reinforce_ratio    : seuil pour envoyer renforts défense
    0.5,   # W[13] neutral_ships_cap  : max ships neutres attaquables / nos ships
]


def W(idx):
    weights = WEIGHTS if WEIGHTS is not None else DEFAULT_W
    return weights[idx]


# ── Physique ─────────────────────────────────────────────────────────────────

def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def is_orbiting(px, py, pradius):
    return dist(px, py, SUN_X, SUN_Y) + pradius < ROT_LIMIT


def predict_pos(px, py, pradius, angular_vel, turns):
    """Position planète dans `turns` tours."""
    if not is_orbiting(px, py, pradius):
        return px, py
    dx, dy = px - SUN_X, py - SUN_Y
    r = math.hypot(dx, dy)
    a = math.atan2(dy, dx) + angular_vel * turns
    return SUN_X + r * math.cos(a), SUN_Y + r * math.sin(a)


def segment_min_dist_to_sun(x1, y1, x2, y2):
    """Distance min entre le soleil et le segment (x1,y1)→(x2,y2)."""
    seg_dx, seg_dy = x2 - x1, y2 - y1
    lsq = seg_dx * seg_dx + seg_dy * seg_dy
    if lsq < 1e-9:
        return dist(x1, y1, SUN_X, SUN_Y)
    t = max(0.0, min(1.0, ((SUN_X - x1) * seg_dx + (SUN_Y - y1) * seg_dy) / lsq))
    return dist(x1 + t * seg_dx, y1 + t * seg_dy, SUN_X, SUN_Y)


def safe_angle(sx, sy, tx, ty):
    """Angle évitant le soleil. Si obstacle, utilise waypoint latéral."""
    direct = math.atan2(ty - sy, tx - sx)
    d = dist(sx, sy, tx, ty)
    ex, ey = sx + math.cos(direct) * d, sy + math.sin(direct) * d

    if segment_min_dist_to_sun(sx, sy, ex, ey) > SUN_RADIUS + 1.5:
        return direct  # chemin libre

    # Trouver waypoint latéral autour du soleil
    perp = direct + math.pi / 2
    wp_r = SUN_RADIUS * (1.0 + W(9) * 8)  # rayon du waypoint

    for sign in [1, -1]:
        wp_x = SUN_X + wp_r * math.cos(direct + sign * math.pi / 2)
        wp_y = SUN_Y + wp_r * math.sin(direct + sign * math.pi / 2)
        # Vérifier les deux segments
        ok1 = segment_min_dist_to_sun(sx, sy, wp_x, wp_y) > SUN_RADIUS + 1.0
        ok2 = segment_min_dist_to_sun(wp_x, wp_y, tx, ty) > SUN_RADIUS + 1.0
        if ok1 and ok2:
            # Retourner angle vers waypoint
            return math.atan2(wp_y - sy, wp_x - sx)

    # Fallback: dévier progressivement
    for delta in [0.3, 0.6, 0.9, 1.2, 1.5]:
        for sign in [1, -1]:
            a = direct + sign * delta
            ex2 = sx + math.cos(a) * d
            ey2 = sy + math.sin(a) * d
            if segment_min_dist_to_sun(sx, sy, ex2, ey2) > SUN_RADIUS + 1.0:
                return a
    return direct


# ── Analyse état du jeu ──────────────────────────────────────────────────────

def ships_by_player(planets, fleets):
    """Total ships (planètes + flottes) par joueur."""
    total = defaultdict(int)
    for p in planets:
        if p[1] >= 0:
            total[p[1]] += p[5]
    for f in fleets:
        if f[1] >= 0:
            total[f[1]] += f[6]
    return total


def threat_to_planet(pid, px, py, fleets, me):
    """Ships ennemis arrivant sur la planète, pondérés par ETA."""
    threat = 0
    for f in fleets:
        fid, fowner, fx, fy, fangle, ffrom, fships = f
        if fowner == me:
            continue
        to_angle = math.atan2(py - fy, px - fx)
        diff = abs(((fangle - to_angle + math.pi) % (2 * math.pi)) - math.pi)
        d = dist(fx, fy, px, py)
        if diff < 0.35 and d < 70:
            eta = d / fleet_speed(fships)
            # Plus la flotte est proche, plus la menace est urgente
            urgency = 1.0 / (1.0 + eta * W(11))
            threat += fships * urgency
    return threat


def friendly_incoming(pid, px, py, fleets, me):
    """Ships alliés déjà en route vers cette planète."""
    incoming = 0
    for f in fleets:
        fid, fowner, fx, fy, fangle, ffrom, fships = f
        if fowner != me:
            continue
        to_angle = math.atan2(py - fy, px - fx)
        diff = abs(((fangle - to_angle + math.pi) % (2 * math.pi)) - math.pi)
        if diff < 0.35 and dist(fx, fy, px, py) < 70:
            incoming += fships
    return incoming


# ── Scoring cibles ───────────────────────────────────────────────────────────

def score_target(sx, sy, s_ships, tgt, tgt_pred_x, tgt_pred_y,
                 me, player_totals, is_comet):
    tid, towner, tx, ty, tr, t_ships, t_prod = tgt
    d = dist(sx, sy, tgt_pred_x, tgt_pred_y)
    if d < 0.1:
        return -1e9

    # Gain estimé
    if towner == -1:  # neutre
        gain = t_prod * W(2) + 5
        priority = W(0)
    else:  # ennemi
        gain = t_prod * W(2) * 0.8 + t_ships * 0.3
        priority = 1.0

        # Pénaliser attaque du leader (stratégie 4P)
        my_ships = player_totals.get(me, 1)
        tgt_ships = player_totals.get(towner, 0)
        if tgt_ships > my_ships * 1.5:
            priority *= (1.0 - W(7))  # evite d'attaquer le dominant

        # Bonus ennemi faible
        if t_ships < s_ships * 0.3:
            priority *= (1.0 + W(8))

    if is_comet:
        priority *= W(1)

    cost = t_ships + 5
    score = priority * gain / (cost + 1) - d * W(3)
    return score


# ── Agent principal ──────────────────────────────────────────────────────────

def agent(obs, config=None):
    planets     = obs.get("planets", [])
    fleets      = obs.get("fleets", [])
    me          = obs.get("player", 0)
    ang_vel     = obs.get("angular_velocity", 0.0)
    comet_ids   = set(obs.get("comet_planet_ids", []))
    step        = obs.get("step", 0)

    my_planets  = [p for p in planets if p[1] == me]
    if not my_planets:
        return []

    player_totals = ships_by_player(planets, fleets)
    my_total      = player_totals.get(me, 0)
    max_enemy     = max((v for k, v in player_totals.items() if k != me), default=0)

    # Mode défense si on domine largement (économiser ships)
    dominant = my_total > max_enemy * (1.0 + W(10)) and my_total > 500

    # Ships déjà engagés sur chaque cible (éviter doublons)
    committed = defaultdict(int)

    moves = []

    # ── Étape 1 : défense urgente ────────────────────────────────────────────
    for p in my_planets:
        pid, _, px, py, pr, p_ships, p_prod = p
        threat = threat_to_planet(pid, px, py, fleets, me)
        if threat <= 0:
            continue
        incoming_ally = friendly_incoming(pid, px, py, fleets, me)
        net_threat = threat - incoming_ally - p_ships
        if net_threat <= 0:
            continue

        # Chercher planète alliée la plus proche pour envoyer renforts
        donors = sorted(
            [q for q in my_planets if q[0] != pid and q[5] > 15],
            key=lambda q: dist(q[2], q[3], px, py)
        )
        for donor in donors[:2]:
            did, _, dx, dy, dr, d_ships, d_prod = donor
            can_send = int(d_ships * (1.0 - W(4)))
            if can_send < 5:
                continue
            reinforce = min(can_send, int(net_threat * W(12)))
            if reinforce < 5:
                continue
            angle = safe_angle(dx, dy, px, py)
            moves.append([did, angle, reinforce])
            committed[pid] += reinforce
            net_threat -= reinforce
            if net_threat <= 0:
                break

    # ── Étape 2 : attaque / expansion ───────────────────────────────────────
    for src in my_planets:
        sid, _, sx, sy, sr, s_ships, s_prod = src

        # Ships disponibles (après réserve défense)
        reserved = int(s_ships * W(4))
        available = s_ships - reserved
        if available < 8:
            continue

        best_score  = -1e18
        best_tgt    = None
        best_pred   = None
        best_needed = 0

        for tgt in planets:
            tid, towner, tx, ty, tr, t_ships, t_prod = tgt
            if towner == me:
                continue

            # Ne pas attaquer neutre trop fort si on est en mode dominant
            if dominant and towner == -1 and t_ships > s_ships * W(13):
                continue

            # Prédire position à l'arrivée
            ships_est = max(int(t_ships * W(5)), 8)
            arrival   = dist(sx, sy, tx, ty) / fleet_speed(ships_est)
            pred_x, pred_y = predict_pos(tx, ty, tr, ang_vel, arrival)

            # Ships nécessaires = garrison actuelle + production pendant trajet + buffer
            needed = int(t_ships + t_prod * arrival + 5)
            needed = int(needed * W(5))  # ratio sécurité

            # Déduire ships déjà engagés vers cette cible
            already = committed[tid]
            effective_needed = max(needed - already, 5)

            if available < effective_needed:
                continue

            sc = score_target(sx, sy, s_ships, tgt, pred_x, pred_y,
                              me, player_totals, tid in comet_ids)
            if sc > best_score:
                best_score  = sc
                best_tgt    = tgt
                best_pred   = (pred_x, pred_y)
                best_needed = effective_needed

        if best_tgt is None:
            continue

        tid = best_tgt[0]
        send = min(int(available * W(6)), available)
        send = max(send, best_needed)
        send = min(send, s_ships - 1)  # garder au moins 1 ship
        if send < 5:
            continue

        angle = safe_angle(sx, sy, best_pred[0], best_pred[1])
        moves.append([sid, angle, send])
        committed[tid] += send

    return moves
