"""Interactive local Orbit Wars simulator.

Controls:
- left click own planet: select source
- left click another planet: queue action
- right click / Backspace: remove last queued action
- Space: apply queued human actions and V9 bot actions
- +/-: change send fraction
- N: new map
- R: reset same seed
- B: toggle bot preview
- Esc/Q: quit
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Optional, Tuple

import pygame

try:
    from . import engine
except ImportError:  # Direct launch: python3 local_simulator/app.py
    import engine


WIDTH, HEIGHT = 1280, 860
BOARD_LEFT, BOARD_TOP = 44, 46
BOARD_SIZE = 760
PANEL_X = BOARD_LEFT + BOARD_SIZE + 34
BG = (5, 8, 15)
BOARD_BG = (8, 13, 24)
GRID = (24, 40, 58)
WHITE = (236, 242, 247)
MUTED = (137, 153, 170)
DIM = (74, 91, 111)
HUMAN = (66, 178, 255)
BOT = (255, 89, 96)
NEUTRAL = (172, 169, 153)
FLEET_HUMAN = (120, 218, 255)
FLEET_BOT = (255, 134, 124)
PENDING = (255, 211, 82)
BOT_PREVIEW = (255, 111, 111)
SUN = (255, 178, 55)
SUN_HOT = (255, 231, 122)
PANEL = (12, 20, 33)


def make_starfield() -> pygame.Surface:
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill(BG)
    rng = pygame.math.Vector2
    for i in range(220):
        x = (i * 197 + 53) % WIDTH
        y = (i * 113 + 91) % HEIGHT
        brightness = 70 + (i * 37) % 150
        radius = 1 if i % 9 else 2
        color = (brightness, brightness, min(255, brightness + 25))
        pygame.draw.circle(surface, color, (x, y), radius)
    for i in range(5):
        cx = 150 + i * 240
        cy = 120 + ((i * 173) % 560)
        pygame.draw.circle(surface, (9, 21, 39), (cx, cy), 120 + 25 * (i % 2))
    return surface.convert()


def board_to_screen(x: float, y: float) -> Tuple[int, int]:
    return int(BOARD_LEFT + x / 100.0 * BOARD_SIZE), int(BOARD_TOP + y / 100.0 * BOARD_SIZE)


def screen_to_board(pos: Tuple[int, int]) -> Tuple[float, float]:
    x, y = pos
    return ((x - BOARD_LEFT) / BOARD_SIZE * 100.0, (y - BOARD_TOP) / BOARD_SIZE * 100.0)


def planet_at(state, pos: Tuple[int, int]) -> Optional[int]:
    bx, by = screen_to_board(pos)
    best = None
    best_d = 1.0e9
    for p in state.planets:
        px, py = float(p[engine.sim.P_X]), float(p[engine.sim.P_Y])
        rr = max(float(p[engine.sim.P_R]) + 1.2, 2.3)
        d = math.hypot(bx - px, by - py)
        if d <= rr and d < best_d:
            best = int(p[engine.sim.P_ID])
            best_d = d
    return best


def draw_text(surface, font, text, x, y, color=WHITE):
    surface.blit(font.render(text, True, color), (x, y))


def lighten(color, amount=45):
    return tuple(min(255, int(c + amount)) for c in color)


def draw_glow_circle(surface, center, radius, color, rings=4):
    for i in range(rings, 0, -1):
        alpha_color = tuple(max(0, int(c * (0.22 + 0.12 * i))) for c in color)
        pygame.draw.circle(surface, alpha_color, center, radius + i * 5, 1)


def draw_arrow(surface, start, end, color, width=2):
    pygame.draw.line(surface, tuple(max(0, c // 3) for c in color), start, end, width + 5)
    pygame.draw.line(surface, color, start, end, width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    head = 11
    for off in (2.55, -2.55):
        p = (int(end[0] - math.cos(angle + off) * head), int(end[1] - math.sin(angle + off) * head))
        pygame.draw.line(surface, color, end, p, width)


def action_endpoint(state, action):
    src = engine.planet_by_id(state, int(action[0]))
    if src is None:
        return None
    sx, sy = board_to_screen(float(src[engine.sim.P_X]), float(src[engine.sim.P_Y]))
    angle = float(action[1])
    length = 170
    return (sx, sy), (int(sx + math.cos(angle) * length), int(sy + math.sin(angle) * length))


def draw_board(surface, state, selected_id, pending_actions, bot_preview, show_bot_preview):
    board_rect = pygame.Rect(BOARD_LEFT, BOARD_TOP, BOARD_SIZE, BOARD_SIZE)
    pygame.draw.rect(surface, BOARD_BG, board_rect, border_radius=24)
    pygame.draw.rect(surface, (38, 62, 88), board_rect, 2, border_radius=24)
    pygame.draw.rect(surface, (120, 170, 210), board_rect.inflate(10, 10), 1, border_radius=28)

    for i in range(0, 101, 10):
        x = BOARD_LEFT + i / 100 * BOARD_SIZE
        y = BOARD_TOP + i / 100 * BOARD_SIZE
        pygame.draw.line(surface, GRID, (x, BOARD_TOP), (x, BOARD_TOP + BOARD_SIZE), 1)
        pygame.draw.line(surface, GRID, (BOARD_LEFT, y), (BOARD_LEFT + BOARD_SIZE, y), 1)

    sun_x, sun_y = board_to_screen(engine.sim.CENTER, engine.sim.CENTER)
    sun_r = int(engine.sim.SUN_RADIUS / 100.0 * BOARD_SIZE)
    for r, c in ((sun_r + 42, (52, 30, 17)), (sun_r + 28, (86, 47, 19)), (sun_r + 14, (140, 73, 25))):
        pygame.draw.circle(surface, c, (sun_x, sun_y), r)
    pygame.draw.circle(surface, SUN, (sun_x, sun_y), sun_r)
    pygame.draw.circle(surface, SUN_HOT, (sun_x - 10, sun_y - 12), max(5, sun_r // 4))
    pygame.draw.circle(surface, (255, 112, 49), (sun_x, sun_y), sun_r, 3)
    pygame.draw.circle(surface, (85, 55, 40), (sun_x, sun_y), int((engine.sim.SUN_RADIUS + 4.0) / 100.0 * BOARD_SIZE), 1)

    center = board_to_screen(engine.sim.CENTER, engine.sim.CENTER)
    for orbit_r in (18, 28, 38, 50):
        pygame.draw.circle(surface, (28, 48, 66), center, int(orbit_r / 100.0 * BOARD_SIZE), 1)

    for p in state.planets:
        dx = float(p[engine.sim.P_X]) - engine.sim.CENTER
        dy = float(p[engine.sim.P_Y]) - engine.sim.CENTER
        orbit_r = math.hypot(dx, dy)
        if orbit_r + float(p[engine.sim.P_R]) < engine.sim.ROTATION_RADIUS_LIMIT:
            pygame.draw.circle(surface, (20, 35, 52), center, int(orbit_r / 100.0 * BOARD_SIZE), 1)

    if show_bot_preview:
        for action in bot_preview:
            pts = action_endpoint(state, action)
            if pts:
                draw_arrow(surface, pts[0], pts[1], BOT_PREVIEW, 2)
    for action in pending_actions:
        pts = action_endpoint(state, action)
        if pts:
            draw_arrow(surface, pts[0], pts[1], PENDING, 3)

    for fleet in state.fleets:
        owner = int(fleet[engine.sim.F_OWNER])
        color = FLEET_HUMAN if owner == 0 else FLEET_BOT
        x, y = board_to_screen(float(fleet[engine.sim.F_X]), float(fleet[engine.sim.F_Y]))
        ships = int(fleet[engine.sim.F_SHIPS])
        radius = max(3, min(9, int(3 + math.log(max(ships, 1), 2))))
        angle = float(fleet[engine.sim.F_ANGLE])
        tail = (int(x - math.cos(angle) * (radius + 9)), int(y - math.sin(angle) * (radius + 9)))
        pygame.draw.line(surface, tuple(max(0, c // 2) for c in color), tail, (x, y), 4)
        pygame.draw.circle(surface, tuple(max(0, c // 3) for c in color), (x, y), radius + 5)
        pygame.draw.circle(surface, color, (x, y), radius)
        pygame.draw.circle(surface, (8, 12, 18), (x, y), radius, 1)

    font = pygame.font.Font(None, 22)
    small = pygame.font.Font(None, 18)
    for p in state.planets:
        owner = int(p[engine.sim.P_OWNER])
        color = HUMAN if owner == 0 else BOT if owner == 1 else NEUTRAL
        x, y = board_to_screen(float(p[engine.sim.P_X]), float(p[engine.sim.P_Y]))
        radius = max(13, int(float(p[engine.sim.P_R]) / 100.0 * BOARD_SIZE * 1.9))
        draw_glow_circle(surface, (x, y), radius, color, 3)
        pygame.draw.circle(surface, tuple(max(0, c // 3) for c in color), (x, y), radius + 5)
        pygame.draw.circle(surface, color, (x, y), radius)
        pygame.draw.circle(surface, lighten(color, 35), (x - max(2, radius // 4), y - max(2, radius // 4)), max(3, radius // 4))
        pygame.draw.circle(surface, (5, 8, 13), (x, y), radius, 2)
        if selected_id == int(p[engine.sim.P_ID]):
            pygame.draw.circle(surface, PENDING, (x, y), radius + 8, 3)
        ships = str(int(p[engine.sim.P_SHIPS]))
        prod = f"+{int(p[engine.sim.P_PROD])}"
        surface.blit(font.render(ships, True, (2, 6, 10)), (x - font.size(ships)[0] // 2, y - 9))
        surface.blit(small.render(prod, True, (2, 6, 10)), (x - small.size(prod)[0] // 2, y + 9))

    label_font = pygame.font.Font(None, 18)
    draw_text(surface, label_font, "HUMAIN", BOARD_LEFT + 16, BOARD_TOP + 14, HUMAN)
    draw_text(surface, label_font, "BOT V9 BEST", BOARD_LEFT + BOARD_SIZE - 104, BOARD_TOP + 14, BOT)


def draw_panel(surface, state, selected_id, send_fraction, pending_actions, bot_preview, last_result, show_bot_preview):
    panel_rect = pygame.Rect(PANEL_X - 18, BOARD_TOP, WIDTH - PANEL_X - 26, BOARD_SIZE)
    pygame.draw.rect(surface, PANEL, panel_rect, border_radius=22)
    pygame.draw.rect(surface, (42, 65, 88), panel_rect, 2, border_radius=22)
    title = pygame.font.Font(None, 38)
    font = pygame.font.Font(None, 24)
    small = pygame.font.Font(None, 20)
    x = PANEL_X
    y = BOARD_TOP + 24
    draw_text(surface, title, "Orbit Wars", x, y)
    y += 32
    draw_text(surface, small, "Human vs V9 best submission", x, y, MUTED)
    y += 34
    scores = engine.scores(state)
    status = "En cours"
    if engine.is_terminal(state):
        w = engine.winner(state)
        status = "Egalite" if w < 0 else ("Humain gagne" if w == 0 else "Bot gagne")

    stat_rows = [
        ("TOUR", f"{state.step}/{engine.sim.TOTAL_TURNS}", WHITE),
        ("HUMAIN", str(scores[0]), HUMAN),
        ("BOT V9", str(scores[1]), BOT),
        ("STATUT", status, WHITE),
        ("FLOTTES", str(len(state.fleets)), WHITE),
        ("ENVOI", f"{int(send_fraction * 100)}%", PENDING),
    ]
    col_w = 132
    for i, (label, value, color) in enumerate(stat_rows):
        rx = x + (i % 2) * col_w
        ry = y + (i // 2) * 58
        pygame.draw.rect(surface, (17, 29, 45), (rx, ry, 118, 48), border_radius=12)
        draw_text(surface, small, label, rx + 10, ry + 7, DIM)
        draw_text(surface, font, value, rx + 10, ry + 24, color)
    y += 190

    draw_text(surface, font, "Selection", x, y, PENDING)
    y += 26
    draw_text(surface, small, f"Source: {selected_id if selected_id is not None else '-'}", x, y, MUTED)
    y += 22
    draw_text(surface, small, f"Actions en attente: {len(pending_actions)}", x, y, MUTED)
    y += 22
    draw_text(surface, small, f"Preview bot: {'ON' if show_bot_preview else 'OFF'} ({len(bot_preview)})", x, y, MUTED)
    y += 34

    draw_text(surface, font, "Controles", x, y, PENDING)
    y += 30
    controls = [
        "Clic ta planete: selection",
        "Clic cible: preparer action",
        "Clic droit / Backspace: annuler derniere",
        "Espace: valider le tour",
        "+ / -: changer la fraction",
        "B: preview bot  N: nouvelle carte",
        "R: reset seed",
        "Esc / Q: quitter",
    ]
    for row in controls:
        draw_text(surface, small, row, x, y, MUTED)
        y += 22

    y += 12
    draw_text(surface, font, "Actions", x, y, PENDING)
    y += 28
    for label, actions, color in (("Humain", pending_actions, PENDING), ("Bot", bot_preview if show_bot_preview else [], BOT_PREVIEW)):
        draw_text(surface, small, f"{label}:", x, y, color)
        y += 21
        if not actions:
            draw_text(surface, small, "-", x + 14, y, MUTED)
            y += 21
        for action in actions[:6]:
            draw_text(surface, small, f"{int(action[0])} angle={float(action[1]):+.2f} ships={int(action[2])}", x + 14, y, MUTED)
            y += 21

    if last_result:
        y += 10
        draw_text(surface, font, "Dernier tour", x, y, PENDING)
        y += 28
        draw_text(surface, small, f"Humain: {len(last_result.human_actions)} actions", x, y, MUTED)
        y += 21
        draw_text(surface, small, f"Bot: {len(last_result.bot_actions)} actions", x, y, MUTED)


def run(seed: int, neutral_pairs: int) -> None:
    pygame.init()
    pygame.display.set_caption("Orbit Wars Local Simulator - Human vs V9 best")
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    background = make_starfield()
    clock = pygame.time.Clock()

    current_seed = seed
    state = engine.new_state(seed=current_seed, neutral_pairs=neutral_pairs)
    selected_id = None
    send_fraction = 0.50
    pending_actions: List[List[float]] = []
    bot_preview = engine.bot_actions(state)
    show_bot_preview = True
    last_result = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE and not engine.is_terminal(state):
                    state, last_result = engine.advance_turn(state, pending_actions)
                    selected_id = None
                    pending_actions = []
                    bot_preview = engine.bot_actions(state) if not engine.is_terminal(state) else []
                elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                    if pending_actions:
                        pending_actions.pop()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    send_fraction = min(1.0, send_fraction + 0.05)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    send_fraction = max(0.05, send_fraction - 0.05)
                elif event.key == pygame.K_b:
                    show_bot_preview = not show_bot_preview
                elif event.key == pygame.K_n:
                    current_seed += 1
                    state = engine.new_state(seed=current_seed, neutral_pairs=neutral_pairs)
                    selected_id = None
                    pending_actions = []
                    bot_preview = engine.bot_actions(state)
                    last_result = None
                elif event.key == pygame.K_r:
                    state = engine.new_state(seed=current_seed, neutral_pairs=neutral_pairs)
                    selected_id = None
                    pending_actions = []
                    bot_preview = engine.bot_actions(state)
                    last_result = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    if pending_actions:
                        pending_actions.pop()
                    continue
                if event.button != 1 or engine.is_terminal(state):
                    continue
                pid = planet_at(state, event.pos)
                if pid is None:
                    selected_id = None
                    continue
                p = engine.planet_by_id(state, pid)
                if p is not None and int(p[engine.sim.P_OWNER]) == engine.HUMAN_PLAYER:
                    selected_id = pid
                elif selected_id is not None:
                    action = engine.make_human_action(state, selected_id, pid, send_fraction)
                    if action is not None:
                        pending_actions.append(action)
                        pending_actions = engine.sanitize_actions(state, engine.HUMAN_PLAYER, pending_actions)

        surface.blit(background, (0, 0))
        draw_board(surface, state, selected_id, pending_actions, bot_preview, show_bot_preview)
        draw_panel(surface, state, selected_id, send_fraction, pending_actions, bot_preview, last_result, show_bot_preview)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Human vs V9 best local Orbit Wars simulator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neutral-pairs", type=int, default=8)
    args = parser.parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    run(seed=args.seed, neutral_pairs=args.neutral_pairs)


if __name__ == "__main__":
    main()
