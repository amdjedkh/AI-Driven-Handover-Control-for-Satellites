import numpy as np
import pandas as pd

# Parameters
NUM_SATELLITES = 50
SIMULATION_STEPS = 20000
USER_POSITION = np.array([0, -1])  # fixed user at bottom
SAT_ALTITUDE = 5                   # "height" of satellites
HANDOVER_THRESHOLD = 0.01        # hysteresis threshold

# Initialize satellites with random positions and speeds
satellites = [
    {
        "id": i,
        "pos": np.array([np.random.uniform(-10, 10), SAT_ALTITUDE]),
        "speed": np.random.uniform(-0.05, 0.05)  # some move left, some move right
    }
    for i in range(NUM_SATELLITES)
]

# Dataset containers
records = []
current_sat = None

def signal_strength(user, sat):
    """Inverse distance as signal strength (normalized)."""
    dist = np.linalg.norm(user - sat)
    return 1.0 / (1.0 + dist)

# Simulation loop
for t in range(SIMULATION_STEPS):
    # Move satellites
    for sat in satellites:
        sat["pos"][0] += sat["speed"]

        # Reset satellite if it drifts too far
        if sat["pos"][0] < -15 or sat["pos"][0] > 15:
            sat["pos"][0] = np.random.uniform(-10, 10)
            sat["pos"][1] = SAT_ALTITUDE + np.random.uniform(-2, 2)
            sat["speed"] = np.random.uniform(-0.05, 0.05)

    # Calculate signal strengths
    strengths = [signal_strength(USER_POSITION, sat["pos"]) for sat in satellites]

    # Best satellite = max signal
    best_sat = np.argmax(strengths)

    # Detect handover
    handover = 0
    if current_sat is None:
        current_sat = best_sat
    elif current_sat != best_sat:
        if strengths[best_sat] - strengths[current_sat] > HANDOVER_THRESHOLD:
            handover = 1
            current_sat = best_sat

    # Save record
    record = {
        "time": t,
        "user_x": USER_POSITION[0],
        "user_y": USER_POSITION[1],
        **{f"sat{i}_x": satellites[i]["pos"][0] for i in range(NUM_SATELLITES)},
        **{f"sat{i}_y": satellites[i]["pos"][1] for i in range(NUM_SATELLITES)},
        **{f"sat{i}_strength": strengths[i] for i in range(NUM_SATELLITES)},
        "connected_sat": current_sat,
        "handover": handover
    }
    records.append(record)

# Save to CSV
df = pd.DataFrame(records)
df.to_csv("handover_dataset.csv", index=False)

# Show dataset balance
print("✅ Dataset generated and saved as handover_dataset.csv")
print(df.head())
print("\nHandover value counts:")
print(df["handover"].value_counts())
