import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("handover_dataset.csv")

# Features (exclude time, connected_sat, handover → labels)
feature_cols = [col for col in df.columns if col not in ["time", "connected_sat", "handover"]]
X = df[feature_cols]
y = df["handover"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump((model, feature_cols), "handover_model.pkl")
print("✅ Model + feature list saved as handover_model.pkl")
