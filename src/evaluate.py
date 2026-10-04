from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.inference import SentryLiteInference


EVALUATION_ROOT = Path("evaluation")
SAFE_DIRECTORY = EVALUATION_ROOT / "safe"
RISKY_DIRECTORY = EVALUATION_ROOT / "risky"

THRESHOLD = 0.50


@dataclass
class EvaluationResult:
    path: Path
    expected: int
    predicted: int
    probability: float


def collect_files(
    directory: Path,
) -> list[Path]:

    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*.py")
        if path.is_file()
    )


def evaluate_directory(
    engine: SentryLiteInference,
    directory: Path,
    expected_label: int,
) -> list[EvaluationResult]:

    results: list[EvaluationResult] = []

    for path in collect_files(directory):

        probability, _ = engine.analyze_file(path)

        predicted = (
            1
            if probability >= THRESHOLD
            else 0
        )

        results.append(
            EvaluationResult(
                path=path,
                expected=expected_label,
                predicted=predicted,
                probability=probability,
            )
        )

    return results


def calculate_metrics(
    results: list[EvaluationResult],
) -> dict[str, float | int]:

    tp = sum(
        result.expected == 1
        and result.predicted == 1
        for result in results
    )

    tn = sum(
        result.expected == 0
        and result.predicted == 0
        for result in results
    )

    fp = sum(
        result.expected == 0
        and result.predicted == 1
        for result in results
    )

    fn = sum(
        result.expected == 1
        and result.predicted == 0
        for result in results
    )

    total = len(results)

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "total": total,
        "correct": tp + tn,
        "wrong": fp + fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_result(
    result: EvaluationResult,
) -> None:

    expected = (
        "RISKY"
        if result.expected
        else "SAFE"
    )

    predicted = (
        "RISKY"
        if result.predicted
        else "SAFE"
    )

    status = (
        "PASS"
        if result.expected == result.predicted
        else "FAIL"
    )

    print(
        f"[{status}] "
        f"{result.path} | "
        f"expected={expected} | "
        f"predicted={predicted} | "
        f"score={result.probability * 100:.2f}%"
    )


def main() -> int:

    engine = SentryLiteInference()

    safe_results = evaluate_directory(
        engine,
        SAFE_DIRECTORY,
        expected_label=0,
    )

    risky_results = evaluate_directory(
        engine,
        RISKY_DIRECTORY,
        expected_label=1,
    )

    results = (
        safe_results
        + risky_results
    )

    if not results:
        print(
            "No evaluation files found.\n\n"
            "Expected directories:\n"
            "  evaluation/safe/\n"
            "  evaluation/risky/"
        )

        return 1

    print()
    print("SentryLite Evaluation")
    print("=" * 70)

    for result in results:
        print_result(result)

    metrics = calculate_metrics(results)

    print()
    print("RESULTS")
    print("=" * 70)

    print(
        f"Samples:        {metrics['total']}"
    )

    print(
        f"Correct:        {metrics['correct']}"
    )

    print(
        f"Wrong:          {metrics['wrong']}"
    )

    print()

    print(
        f"Accuracy:       "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision:      "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:         "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1:             "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print()

    print(
        f"True positives:  {metrics['tp']}"
    )

    print(
        f"True negatives:  {metrics['tn']}"
    )

    print(
        f"False positives: {metrics['fp']}"
    )

    print(
        f"False negatives: {metrics['fn']}"
    )

    failures = [
        result
        for result in results
        if result.expected != result.predicted
    ]

    if failures:
        print()
        print("FAILED SAMPLES")
        print("=" * 70)

        for result in failures:
            print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())