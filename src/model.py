from __future__ import annotations

import numpy as np


class LogisticRegression:
    """
    Tiny binary classifier implemented with NumPy.

    Input:
        Feature vector

    Output:
        Probability from 0.0 to 1.0

    0 -> SAFE
    1 -> RISKY
    """

    def __init__(self, feature_count: int) -> None:
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than zero")

        self.feature_count = feature_count

        self.weights = np.zeros(
            feature_count,
            dtype=np.float32,
        )

        self.bias = np.float32(0.0)

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(
            values,
            -30.0,
            30.0,
        )

        return 1.0 / (1.0 + np.exp(-values))

    def predict_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        logits = features @ self.weights + self.bias

        return self.sigmoid(logits)

    def predict(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:

        probabilities = self.predict_proba(features)

        return (
            probabilities >= threshold
        ).astype(np.int8)

    def parameter_count(self) -> int:
        return self.weights.size + 1

    def size_bytes(self) -> int:
        return self.weights.nbytes + self.bias.nbytes