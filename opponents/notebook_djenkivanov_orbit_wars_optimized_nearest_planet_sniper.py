"""Auto-extracted from a Kaggle notebook."""

import math
import kaggle_environments.envs.orbit_wars.orbit_wars as ow

fleet_trajectories = []
moving_planets = []
planets_coords = {}
steps = 0


def update_fleet_trajectories(obs):
    fleets = [ow.Fleet(*f) for f in obs.get("fleets", [])]
    for f_t in fleet_trajectories[:]:
        mine_id, angle, nearest_id = f_t
        found = False
        for f in fleets:
            if f.from_planet_id == mine_id and abs(f.angle - angle) < 1e-6:
                found = True
                break

        if not found:
            fleet_trajectories.remove(f_t)


def get_planet_trajectories(p, vel):
    planet_trajectories = []
    angle = math.atan2(p.y - 50, p.x - 50)
    r = math.sqrt((p.x - 50)**2 + (p.y - 50)**2)
    for t in range(60):
        angle_t = angle + vel * t
        x_t = 50 + r * math.cos(angle_t)
        y_t = 50 + r * math.sin(angle_t)
        planet_trajectories.append((x_t, y_t))

    return planet_trajectories
    

def fill_moving_planets(obs):
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    initial_by_id = {i[0]: ow.Planet(*i) for i in obs.get("initial_planets", [])}
    for p in planets:
        i = initial_by_id[p.id]
        if (p.x, p.y) != (i.x, i.y):
            if p.id not in moving_planets:
                moving_planets.append(p.id)


def agent(obs):
    global steps
    moves = []
    if steps < 2:
        steps += 1
        return []
    if steps == 2:
        fill_moving_planets(obs)
        steps = 3   

    update_fleet_trajectories(obs)

    player = obs.get("player", [])
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return []

    for mine in my_planets:
        nearest = None
        min_dist = float('inf')
        for t in targets:
            # remove 1 fleet per target rule in late game to snowball faster
            if len(targets) > 6 and any(already_targeted_planet_id == t.id for _, _, already_targeted_planet_id in fleet_trajectories):
                continue
            dist = math.sqrt((mine.x - t.x)**2 + (mine.y - t.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        base_ships = nearest.ships + 1
        extra_ships = 0

        if mine.ships >= base_ships:
            angle = None
            if nearest.id in moving_planets:
                maxSpeed = 6.0
                planet_trajectories = get_planet_trajectories(nearest, obs.angular_velocity)
                for t, (px, py) in enumerate(planet_trajectories[1:], start=1):
                    if nearest.owner != -1:
                        extra_ships = t * nearest.production
                    if base_ships + extra_ships > mine.ships:
                        continue
                    fleet_speed = 1.0 + (maxSpeed - 1.0) * (math.log(base_ships + extra_ships) / math.log(1000)) ** 1.5
                    angle_t = math.atan2(py - mine.y, px - mine.x)
                    dist = math.sqrt((px - mine.x)**2 + (py - mine.y)**2)
                    turns_to_arrive = math.ceil(dist / fleet_speed)
                    if abs(turns_to_arrive - t) > 1:
                        continue
                    angle = angle_t
                    break
            else:
                angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
                if nearest.owner != -1:
                    maxSpeed = 6.0
                    dist = math.sqrt((nearest.x - mine.x)**2 + (nearest.y - mine.y)**2)
                
                    fleet_speed = 1.0 + (maxSpeed - 1.0) * (math.log(base_ships) / math.log(1000)) ** 1.5
                    turns_to_arrive = math.ceil(dist / fleet_speed)
                    extra_ships = turns_to_arrive * nearest.production
                
                    total_ships = base_ships + extra_ships
                    fleet_speed = 1.0 + (maxSpeed - 1.0) * (math.log(total_ships) / math.log(1000)) ** 1.5  # FIX
                    turns_to_arrive = math.ceil(dist / fleet_speed)
                    extra_ships = turns_to_arrive * nearest.production
                    
            if angle is not None:
                total_ships = base_ships + extra_ships
                moves.append([mine.id, angle, total_ships])
                fleet_trajectories.append((mine.id, angle, nearest.id))

    return moves
