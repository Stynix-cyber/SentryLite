from __future__ import annotations

from pathlib import Path

import numpy as np

from src.dataset import load_dataset
from src.model import LogisticRegression


SEED = 42

LEARNING_RATE = 0.05
MAX_EPOCHS = 3000
PATIENCE = 150

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentrylite_v01.npz"


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    epsilon = 1e-7

    y_pred = np.clip(
        y_pred,
        epsilon,
        1.0 - epsilon,
    )

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1.0 - y_true) * np.log(1.0 - y_pred)
    )

    return float(loss)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    true_positive = int(
        np.sum((y_true == 1) & (y_pred == 1))
    )

    true_negative = int(
        np.sum((y_true == 0) & (y_pred == 0))
    )

    false_positive = int(
        np.sum((y_true == 0) & (y_pred == 1))
    )

    false_negative = int(
        np.sum((y_true == 1) & (y_pred == 0))
    )

    total = len(y_true)

    accuracy = (
        (true_positive + true_negative) / total
        if total
        else 0.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positive),
        "tn": float(true_negative),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(SEED)

    safe_indices = np.where(y == 0)[0]
    risky_indices = np.where(y == 1)[0]

    rng.shuffle(safe_indices)
    rng.shuffle(risky_indices)

    def split_class(
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        count = len(indices)

        train_end = int(count * TRAIN_RATIO)

        validation_end = train_end + int(
            count * VALIDATION_RATIO
        )

        return (
            indices[:train_end],
            indices[train_end:validation_end],
            indices[validation_end:],
        )

    safe_train, safe_validation, safe_test = split_class(
        safe_indices
    )

    risky_train, risky_validation, risky_test = split_class(
        risky_indices
    )

    train_indices = np.concatenate(
        [safe_train, risky_train]
    )

    validation_indices = np.concatenate(
        [safe_validation, risky_validation]
    )

    test_indices = np.concatenate(
        [safe_test, risky_test]
    )

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)

    return (
        x[train_indices],
        y[train_indices],
        x[validation_indices],
        y[validation_indices],
        x[test_indices],
        y[test_indices],
    )


def normalize(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    mean = x_train.mean(axis=0)

    std = x_train.std(axis=0)

    # Prevent division by zero.
    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return (
        (x_train - mean) / std,
        (x_validation - mean) / std,
        (x_test - mean) / std,
        mean,
        std,
    )


def save_model(
    model: LogisticRegression,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        MODEL_PATH,
        weights=model.weights,
        bias=model.bias,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


def main() -> None:
    print()
    print("SentryLite Training")
    print("=" * 55)

    dataset = load_dataset()

    if not dataset.samples:
        raise RuntimeError("Dataset is empty.")

    x = np.asarray(
        [
            sample.features
            for sample in dataset.samples
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        [
            sample.label
            for sample in dataset.samples
        ],
        dtype=np.float32,
    )

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = split_dataset(x, y)

    (
        x_train,
        x_validation,
        x_test,
        mean,
        std,
    ) = normalize(
        x_train,
        x_validation,
        x_test,
    )

    print(f"Training samples:   {len(x_train)}")
    print(f"Validation samples: {len(x_validation)}")
    print(f"Test samples:       {len(x_test)}")
    print(f"Features:           {x_train.shape[1]}")
    print()

    model = LogisticRegression(
        feature_count=x_train.shape[1]
    )

    best_validation_loss = float("inf")

    best_weights = model.weights.copy()
    best_bias = np.float32(model.bias)

    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        # -------------------------------------------
        # Forward pass
        # -------------------------------------------

        probabilities = model.predict_proba(
            x_train
        )

        # -------------------------------------------
        # Gradient calculation
        # -------------------------------------------

        error = probabilities - y_train

        weight_gradient = (
            x_train.T @ error
        ) / len(x_train)

        bias_gradient = np.mean(error)

        # -------------------------------------------
        # Gradient descent
        # -------------------------------------------

        model.weights -= (
            LEARNING_RATE
            * weight_gradient.astype(np.float32)
        )

        model.bias -= np.float32(
            LEARNING_RATE * bias_gradient
        )

        # -------------------------------------------
        # Validation
        # -------------------------------------------

        validation_probabilities = (
            model.predict_proba(x_validation)
        )

        validation_loss = binary_cross_entropy(
            y_validation,
            validation_probabilities,
        )

        if validation_loss < (
            best_validation_loss - 1e-7
        ):
            best_validation_loss = validation_loss

            best_weights = model.weights.copy()
            best_bias = np.float32(model.bias)

            best_epoch = epoch
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 100 == 0:
            train_loss = binary_cross_entropy(
                y_train,
                probabilities,
            )

            predictions = model.predict(
                x_validation
            )

            metrics = calculate_metrics(
                y_validation.astype(np.int8),
                predictions,
            )

            print(
                f"Epoch {epoch:4d} | "
                f"train={train_loss:.5f} | "
                f"val={validation_loss:.5f} | "
                f"F1={metrics['f1']:.3f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print()
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

    # Restore the best validation model.
    model.weights = best_weights
    model.bias = best_bias

    print()
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    # -----------------------------------------------
    # Final test set
    # -----------------------------------------------

    test_probabilities = model.predict_proba(
        x_test
    )

    test_loss = binary_cross_entropy(
        y_test,
        test_probabilities,
    )

    test_predictions = model.predict(
        x_test
    )

    metrics = calculate_metrics(
        y_test.astype(np.int8),
        test_predictions,
    )

    print()
    print("FINAL TEST RESULTS")
    print("=" * 55)

    print(f"Loss:      {test_loss:.6f}")
    print(
        f"Accuracy:  {metrics['accuracy']:.3%}"
    )
    print(
        f"Precision: {metrics['precision']:.3%}"
    )
    print(
        f"Recall:    {metrics['recall']:.3%}"
    )
    print(
        f"F1:        {metrics['f1']:.3%}"
    )

    print()
    print(
        f"TP: {int(metrics['tp'])} | "
        f"TN: {int(metrics['tn'])} | "
        f"FP: {int(metrics['fp'])} | "
        f"FN: {int(metrics['fn'])}"
    )

    save_model(
        model,
        mean,
        std,
    )

    print()
    print(f"Model saved to: {MODEL_PATH}")

    model_size = MODEL_PATH.stat().st_size

    print(
        f"Saved model size: "
        f"{model_size:,} bytes"
    )


if __name__ == "__main__":
    main()