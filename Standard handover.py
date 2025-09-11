import pygame
import math
import sys

# Init
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Satellite Handover Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY = (135, 206, 235)

# Load images (replace with your PNGs)
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

while True:
    screen.fill(SKY)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Draw Earth ground curve
    # Draw Earth ground as a half-sphere
    earth_radius = 700
    earth_center = (WIDTH // 2, HEIGHT + 600)  # Center below the screen
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

    # Find closest satellite
    user_center = (ground_x+40, ground_y+40)
    distances = [math.dist(user_center, c) for c in centers]
    closest = distances.index(min(distances))

    # Handover detection
    if active_sat is None:
        active_sat = closest
    elif closest != active_sat:
        handover_events.append((pygame.time.get_ticks(), active_sat, closest))
        active_sat = closest

    # Draw connection line
    pygame.draw.line(screen, WHITE, user_center, centers[active_sat], 2)

    # Show latest handover
    if handover_events:
        txt = f"Handover: Sat {handover_events[-1][1]} → Sat {handover_events[-1][2]}"
        text_surf = font.render(txt, True, BLACK)
        screen.blit(text_surf, (20, 20))

    pygame.display.flip()
    clock.tick(60)
