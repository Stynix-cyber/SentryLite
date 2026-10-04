from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.features import extract_features
from src.model import LogisticRegression


MODEL_PATH = Path("models/sentrylite_v01.npz")

RISK_THRESHOLD = 0.50


class SentryLiteInference:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
    ) -> None:

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run training first with:\n"
                "python -m src.train"
            )

        model_data = np.load(
            model_path,
            allow_pickle=False,
        )

        weights = np.asarray(
            model_data["weights"],
            dtype=np.float32,
        )

        self.mean = np.asarray(
            model_data["mean"],
            dtype=np.float32,
        )

        self.std = np.asarray(
            model_data["std"],
            dtype=np.float32,
        )

        bias = np.float32(
            model_data["bias"]
        )

        if not (
            len(weights)
            == len(self.mean)
            == len(self.std)
        ):
            raise ValueError(
                "Model file contains incompatible dimensions."
            )

        self.model = LogisticRegression(
            feature_count=len(weights)
        )

        self.model.weights = weights
        self.model.bias = bias

    def analyze_source(
        self,
        source: str,
    ) -> tuple[float, list[float]]:

        features = extract_features(source)

        vector = np.asarray(
            features.as_vector(),
            dtype=np.float32,
        )

        if len(vector) != self.model.feature_count:
            raise ValueError(
                "Feature count does not match model."
            )

        normalized = (
            vector - self.mean
        ) / self.std

        probability = self.model.predict_proba(
            normalized
        )

        return (
            float(probability),
            vector.tolist(),
        )

    def analyze_file(
        self,
        path: Path,
    ) -> tuple[float, list[float]]:

        if path.suffix.lower() != ".py":
            raise ValueError(
                "SentryLite currently supports Python files only."
            )

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return self.analyze_source(source)


def risk_level(probability: float) -> str:
    if probability >= 0.90:
        return "HIGH"

    if probability >= 0.70:
        return "ELEVATED"

    if probability >= 0.50:
        return "REVIEW"

    if probability >= 0.25:
        return "LOW"

    return "MINIMAL"


def print_report(
    path: Path,
    probability: float,
    features: list[float],
) -> None:

    classification = (
        "RISKY"
        if probability >= RISK_THRESHOLD
        else "SAFE"
    )

    print()
    print("SentryLite AI")
    print("=" * 55)
    print(f"File:           {path}")
    print(f"Classification: {classification}")
    print(
        f"Risk score:     "
        f"{probability * 100:.2f}%"
    )
    print(
        f"Risk level:     "
        f"{risk_level(probability)}"
    )

    print()
    print("Feature vector:")
    print(features)

    print()
    print(
        "Note: The score is an ML estimate, "
        "not proof of a vulnerability."
    )


def main() -> int:

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python -m src.inference <file.py>"
        )
        return 1

    path = Path(sys.argv[1])

    if not path.is_file():
        print(
            f"File not found: {path}"
        )
        return 1

    try:
        engine = SentryLiteInference()

        probability, features = (
            engine.analyze_file(path)
        )

    except (
        OSError,
        ValueError,
        SyntaxError,
    ) as exc:

        print(
            f"Analysis failed: {exc}"
        )

        return 1

    print_report(
        path,
        probability,
        features,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())