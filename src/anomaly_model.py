# src/anomaly_model.py
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from pathlib import Path

MODEL_PATH = Path("outputs") / "models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)

class IsolationAnomaly:
    def __init__(self, model_file=None):
        self.model_file = model_file if model_file is not None else MODEL_PATH / "iso_forests.joblib"
        self.model = None

    def train(self, X, n_estimators=200, contamination=0.01, random_state=42):
        """
        X: numpy array shape (n_samples, n_features) — features from normal clips
        contamination: expected fraction of anomalies in training (keep small)
        """
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
        self.model.fit(X)
        joblib.dump(self.model, str(self.model_file))
        return self.model

    def load(self):
        if Path(self.model_file).exists():
            self.model = joblib.load(str(self.model_file))
        else:
            raise FileNotFoundError(f"Model not found at {self.model_file}")

    def score_samples(self, X):
        """
        Returns anomaly scores (the lower, the MORE anomalous for IsolationForest)
        We'll flip sign to produce higher = more anomalous.
        """
        if self.model is None:
            self.load()
        raw = self.model.score_samples(X)  # higher = more normal
        # convert to anomaly score in [0, +inf): anomaly = -raw
        return -raw

    def predict(self, X, threshold=None):
        """
        Return boolean anomaly flags (True == anomaly). If threshold None, use model.predict
        """
        if self.model is None:
            self.load()
        preds = self.model.predict(X)  # -1 for outliers, 1 for inliers
        return (preds == -1)

    def save(self, path=None):
        p = Path(path) if path is not None else self.model_file
        joblib.dump(self.model, str(p))
