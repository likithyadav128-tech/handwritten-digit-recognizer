"""
train_model.py
---------------
One-time training script: downloads MNIST (28x28, ~70k real handwritten
digits) and trains a classifier, saving it as model/digit_classifier.joblib
for app.py to load.

Run once locally (needs internet to download MNIST the first time — it's
cached after that):

    python train_model.py
"""

from pathlib import Path

import joblib
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

print("Downloading MNIST (first run only, cached after)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# Scale pixels to 0-1 — matches the normalization app.py applies to both
# uploaded photos and canvas drawings, so train/inference distributions
# line up.
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print(f"Training on {len(X_train):,} samples...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=30,
    random_state=42,
    verbose=True,
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {acc:.4f}")

Path("model").mkdir(exist_ok=True)
joblib.dump(
    {"model": model, "model_name": "MLP (128,64) — MNIST", "test_accuracy": acc},
    "model/digit_classifier.joblib",
)
print("Saved to model/digit_classifier.joblib")
