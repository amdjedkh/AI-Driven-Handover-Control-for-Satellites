# Satellite Handover Simulation with AI

This repository contains the code and experiments for simulating satellite-to-ground handovers using two different approaches:
1. **Baseline policy**: always connect to the closest satellite.
2. **AI-driven policy**: a Random Forest classifier trained on simulated datasets to predict handover events.

The goal is to reduce unnecessary handovers and maintain a stronger connection quality.

---

## 📂 Project Structure
- `handover_dataset.csv` — Generated dataset (20,000 steps, 50 satellites).
- `handover_model.pkl` — Trained Random Forest model + feature list (saved with `joblib`).
- `comparison.py` — Pygame visualization comparing baseline vs AI.
- `train_model.py` — Training script for Random Forest classifier.
- `simulation_video.mp4` — Full demo video (linked below).

---

## 📊 Results
- **Dataset size:** 20,000 samples, 155 features.
- **Class balance:** 19,897 non-handover (0), 103 handover (1).
- **Model performance (Random Forest, 100 estimators):**

Confusion Matrix:
[[3969 6]
[ 25 0]]

Classification Report:
precision recall f1-score support

       0       0.99      1.00      1.00      3975
       1       0.00      0.00      0.00        25

accuracy                           0.99      4000

macro avg 0.50 0.50 0.50 4000
weighted avg 0.99 0.99 0.99 4000


- Baseline produced frequent handovers.
- AI reduced unnecessary switches but requires further rebalancing for rare handover events.

---

## 🎥 Demo Video
Below is a short preview of the simulation (baseline = red dashed line, AI = white solid line):

![Simulation Preview](figures/preview.gif)

👉 [Watch the full video on GitHub](https://github.com/amdjedkh/AI-Driven-Handover-Control-for-Satellites/tree/master/simulation_video.MOV)

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
2. Run the simulation:

python comparison.py

📝 Notes

Current dataset is highly imbalanced (only ~0.5% positive handover events).

Future improvements include resampling (SMOTE/undersampling) and testing deep learning models.

Results and plots are documented in the internship report.

📜 License

This project is licensed under the MIT License.
