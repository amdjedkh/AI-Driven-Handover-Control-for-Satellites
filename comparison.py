"""
compare_handover_sim.py
Side-by-side demo: Baseline (closest) vs AI (trained model) handover logic.
Requirements:
  pip install pygame pandas joblib numpy
Place satellite.png, ground.png in same folder.
If you saved your model as (model, feature_cols) use that; otherwise model only will load.
"""

import pygame
import math
import sys
import pandas as pd
import joblib
import numpy as np
from typing import List, Tuple

# ----- Load ML model (optional) -----
MODEL_PATH = "handover_model.pkl"
model = None
feature_cols = None
ml_available = False
try:
    loaded = joblib.load(MODEL_PATH)
    # handle both (model, feature_cols) and model-only saved files
    if isinstance(loaded, tuple) and len(loaded) >= 1:
        model = loaded[0]
        if len(loaded) > 1:
            feature_cols = loaded[1]
        ml_available = True
        print(f"Loaded model from {MODEL_PATH}. Feature columns available: {feature_cols is not None}")
    else:
        model = loaded
        ml_available = True
        print(f"Loaded model from {MODEL_PATH} (no feature list).")
except Exception as e:
    print(f"Could not load ML model ({MODEL_PATH}): {e}")
    print("Simulation will run using baseline policy only.")

# ----- Pygame init -----
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Baseline (red) vs AI (white) Satellite Handover")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
bigfont = pygame.font.SysFont("Arial", 22, bold=True)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY = (135, 206, 235)
RED = (220, 60, 60)
GREEN = (34, 139, 34)
YELLOW = (255, 220, 100)

# ----- Load images (replace these filenames with yours) -----
sat_img = pygame.image.load("satellite.png").convert_alpha()
sat_img = pygame.transform.smoothscale(sat_img, (56, 56))
ground_img = pygame.image.load("ground.png").convert_alpha()
ground_img = pygame.transform.smoothscale(ground_img, (120, 120))

# Ground station position
ground_x, ground_y = WIDTH // 2 - 60, HEIGHT - 140

# Satellite definition: start x, y, speed (pixels/frame)
sats = [
    {"x": 1200, "y": 180, "speed": -2.1},  # Sat 0
    {"x": 1600, "y": 230, "speed": -1.6},  # Sat 1
    {"x": 2000, "y": 150, "speed": -1.9},  # Sat 2
    {"x": 2400, "y": 200, "speed": -2.0},  # Sat 3 (new)
    {"x": 2800, "y": 170, "speed": -1.7},  # Sat 4 (new)
]


# Simulation state for baseline and AI
active_baseline = None
active_ai = None

baseline_handover_events: List[Tuple[int,int,int]] = []
ai_handover_events: List[Tuple[int,int,int]] = []

# helper: dashed line draw
def draw_dashed_line(surface, color, start_pos, end_pos, width=2, dash_length=8):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = dash_length
    # Total length
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    num = int(dist / dl)
    for i in range(0, num, 2):
        sx = x1 + dx * i * dl
        sy = y1 + dy * i * dl
        ex = x1 + dx * min(i+1, num) * dl
        ey = y1 + dy * min(i+1, num) * dl
        pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)

# signal strength model (same as used for training)
def signal_strength(user, sat_center):
    dist = math.dist(user, sat_center)
    return 1.0 / (1.0 + dist)

# optional: build feature vector in same order as training feature_cols
def build_feature_vector(user_center, centers):
    features = {}
    features["user_x"] = user_center[0]
    features["user_y"] = user_center[1]
    for i, c in enumerate(centers):
        features[f"sat{i}_x"] = c[0]
        features[f"sat{i}_y"] = c[1]
        features[f"sat{i}_strength"] = signal_strength(user_center, c)
    if feature_cols is not None:
        # ensure order and fill missing columns with zeros if any
        row = [features.get(col, 0.0) for col in feature_cols]
        return pd.DataFrame([row], columns=feature_cols)
    else:
        # fallback: use predictable ordering (sat0_x, sat0_y, sat0_strength, sat1_x,...)
        ordered = []
        for i in range(len(centers)):
            ordered.extend([features[f"sat{i}_x"], features[f"sat{i}_y"], features[f"sat{i}_strength"]])
        # also add user coords at end
        ordered.extend([features["user_x"], features["user_y"]])
        # create columns names to be stable
        cols = []
        for i in range(len(centers)):
            cols += [f"sat{i}_x", f"sat{i}_y", f"sat{i}_strength"]
        cols += ["user_x", "user_y"]
        return pd.DataFrame([ordered], columns=cols)

show_baseline_line = True
show_ai_line = True
show_diff_highlight = True

# main loop
frame = 0
while True:
    frame += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # toggle keys for visual debugging
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:
                show_baseline_line = not show_baseline_line
            if event.key == pygame.K_a:
                show_ai_line = not show_ai_line
            if event.key == pygame.K_d:
                show_diff_highlight = not show_diff_highlight

    # background
    screen.fill(SKY)

    # big earth arc / half-sphere
    earth_radius = 850
    earth_center = (WIDTH // 2, HEIGHT + 700)
    pygame.draw.circle(screen, GREEN, earth_center, earth_radius)

    # ground station
    screen.blit(ground_img, (ground_x, ground_y))

    # move and draw satellites, gather centers
    centers = []
    for sat in sats:
        sat["x"] += sat["speed"]
        if sat["x"] < -120:  # when out of left boundary wrap to right
            sat["x"] = WIDTH + 300 + np.random.randint(0, 300)
            # slight randomize y to create varied sequences
            sat["y"] = 120 + np.random.randint(-30, 60)
        sat_rect = sat_img.get_rect(center=(sat["x"], sat["y"]))
        screen.blit(sat_img, sat_rect)
        centers.append((sat["x"], sat["y"]))

    # user (center of ground station)
    user_center = (ground_x + 60, ground_y + 60)

    # compute strengths and baseline choice (closest)
    strengths = [signal_strength(user_center, c) for c in centers]
    baseline_choice = int(np.argmin([math.dist(user_center, c) for c in centers]))
    ai_prediction = 0
    ai_choice = None

    # baseline handover detection
    if active_baseline is None:
        active_baseline = baseline_choice
    elif baseline_choice != active_baseline:
        baseline_handover_events.append((pygame.time.get_ticks(), active_baseline, baseline_choice))
        active_baseline = baseline_choice

    # AI decision (if model available)
    if ml_available and model is not None:
        X_input = build_feature_vector(user_center, centers)
        try:
            ai_prediction = model.predict(X_input)[0]  # expects 0/1 (handover or not)
        except Exception as e:
            # If model expects different shape/names, try to coerce by columns
            try:
                X_input = X_input.reindex(columns=model.feature_names_in_, fill_value=0)
                ai_prediction = model.predict(X_input)[0]
            except Exception as e2:
                ai_prediction = 0
        # if model predicts handover -> switch to strongest satellite (by strength)
        strongest = int(np.argmax(strengths))
        if active_ai is None:
            active_ai = strongest
        elif ai_prediction == 1 and strongest != active_ai:
            ai_handover_events.append((pygame.time.get_ticks(), active_ai, strongest))
            active_ai = strongest
    else:
        # no ML available -> mirror baseline (so white = baseline)
        if active_ai is None:
            active_ai = baseline_choice
        elif baseline_choice != active_ai:
            ai_handover_events.append((pygame.time.get_ticks(), active_ai, baseline_choice))
            active_ai = baseline_choice

    # draw baseline line (red dashed) and ai line (white solid)
    if show_baseline_line and active_baseline is not None:
        draw_dashed_line(screen, RED, user_center, centers[active_baseline], width=3, dash_length=10)
    if show_ai_line and active_ai is not None:
        pygame.draw.line(screen, WHITE, user_center, centers[active_ai], 4)



    # Draw HUD / stats
    hud_x = 12
    hud_y = 10
    # Titles
    title = bigfont.render("Baseline (red dashed)  —  AI (white solid)", True, BLACK)
    screen.blit(title, (hud_x, hud_y))
    hud_y += 30

    # Handovers count
    baseline_count = len(baseline_handover_events)
    ai_count = len(ai_handover_events)
    txt1 = font.render(f"Baseline handovers: {baseline_count}", True, BLACK)
    txt2 = font.render(f"AI handovers:       {ai_count}", True, BLACK)
    screen.blit(txt1, (hud_x, hud_y)); hud_y += 22
    screen.blit(txt2, (hud_x, hud_y)); hud_y += 26

    # Latest events
    if baseline_handover_events:
        t0, b_from, b_to = baseline_handover_events[-1]
        last_b = font.render(f"Baseline last: {b_from} → {b_to}", True, BLACK)
        screen.blit(last_b, (hud_x, hud_y)); hud_y += 20
    if ai_handover_events:
        t1, a_from, a_to = ai_handover_events[-1]
        last_a = font.render(f"AI last:       {a_from} → {a_to}", True, BLACK)
        screen.blit(last_a, (hud_x, hud_y)); hud_y += 20

    # show model availability
    model_txt = "ML model: available" if ml_available else "ML model: NOT loaded"
    screen.blit(font.render(model_txt, True, BLACK), (hud_x, hud_y))
    hud_y += 20

    # show current choices next to user
    base_choice_text = font.render(f"Baseline -> Sat {active_baseline}", True, RED)
    ai_choice_text = font.render(f"AI -> Sat {active_ai}", True, BLACK)
    screen.blit(base_choice_text, (WIDTH - 240, 14))
    screen.blit(ai_choice_text, (WIDTH - 240, 36))

    # draw small legend icons for satellites with IDs
    for i, c in enumerate(centers):
        label = font.render(f"Sat {i}", True, BLACK)
        screen.blit(label, (c[0] - 10, c[1] - 35))

    pygame.display.flip()
    clock.tick(60)
