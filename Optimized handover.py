import pygame
import math
import sys
import pandas as pd
import joblib
import numpy as np

# Load trained model + feature names
model, feature_cols = joblib.load("handover_model.pkl")

# Init
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI-Powered Satellite Handover Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY = (135, 206, 235)

# Load images
sat_img = pygame.image.load("satellite.png")
sat_img = pygame.transform.scale(sat_img, (50, 50))
ground_img = pygame.image.load("ground.png")
ground_img = pygame.transform.scale(ground_img, (80, 80))

# Ground station position
ground_x, ground_y = WIDTH//2 - 40, HEIGHT - 100

# Satellite positions (moving)
sats = [
    {"x": 200, "y": 200, "speed": -2},
    {"x": 500, "y": 250, "speed": -1.5},
    {"x": 700, "y": 180, "speed": -1.8},
]

active_sat = None
handover_events = []

font = pygame.font.SysFont("Arial", 20)
clock = pygame.time.Clock()

def signal_strength(user, sat):
    """Simple inverse-distance signal strength"""
    dist = math.dist(user, sat)
    return 1.0 / (1.0 + dist)

while True:
    screen.fill(SKY)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Draw Earth ground as a half-sphere
    earth_radius = 700
    earth_center = (WIDTH // 2, HEIGHT + 600)
    pygame.draw.circle(screen, (34, 139, 34), earth_center, earth_radius)

    # Draw ground station
    screen.blit(ground_img, (ground_x, ground_y))

    # Move satellites
    centers = []
    for sat in sats:
        sat["x"] += sat["speed"]
        if sat["x"] < -50:  # wrap around
            sat["x"] = WIDTH + 50
        screen.blit(sat_img, (sat["x"], sat["y"]))
        centers.append((sat["x"]+25, sat["y"]+25))

    # User center
    user_center = (ground_x+40, ground_y+40)

    # Calculate signal strengths
    strengths = [signal_strength(user_center, c) for c in centers]

    # Build ML input
    features = {
        "user_x": user_center[0],
        "user_y": user_center[1],
    }
    for i, c in enumerate(centers):
        features[f"sat{i}_x"] = c[0]
        features[f"sat{i}_y"] = c[1]
        features[f"sat{i}_strength"] = strengths[i]

    # Ensure correct order
    X_input = pd.DataFrame([[features[col] for col in feature_cols]], columns=feature_cols)

    # Predict handover
    prediction = model.predict(X_input)[0]  # 0=no handover, 1=handover



    # Use prediction
    if active_sat is None:
        active_sat = np.argmax(strengths)  # pick strongest first
    elif prediction == 1:
        # If ML predicts handover → switch to strongest
        new_sat = np.argmax(strengths)
        if new_sat != active_sat:
            handover_events.append((pygame.time.get_ticks(), active_sat, new_sat))
            active_sat = new_sat

    # Draw connection line
    pygame.draw.line(screen, WHITE, user_center, centers[active_sat], 2)

    # Show latest handover
    if handover_events:
        txt = f"AI Handover: Sat {handover_events[-1][1]} → Sat {handover_events[-1][2]}"
        text_surf = font.render(txt, True, BLACK)
        screen.blit(text_surf, (20, 20))

    pygame.display.flip()
    clock.tick(60)
