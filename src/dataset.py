from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.features import extract_features


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    features: list[float]
    label: int


@dataclass(frozen=True)
class Dataset:
    samples: list[DatasetSample]

    @property
    def safe_count(self) -> int:
        return sum(sample.label == 0 for sample in self.samples)

    @property
    def risky_count(self) -> int:
        return sum(sample.label == 1 for sample in self.samples)


def load_directory(
    directory: Path,
    label: int,
) -> list[DatasetSample]:

    samples: list[DatasetSample] = []

    for path in sorted(directory.glob("*.py")):
        try:
            source = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            features = extract_features(source)

            samples.append(
                DatasetSample(
                    path=path,
                    features=features.as_vector(),
                    label=label,
                )
            )

        except (SyntaxError, OSError, ValueError) as exc:
            print(f"Skipping {path}: {exc}")

    return samples


def load_dataset(
    root: Path = Path("dataset"),
) -> Dataset:

    safe_directory = root / "safe"
    risky_directory = root / "risky"

    if not safe_directory.is_dir():
        raise FileNotFoundError(
            f"Safe dataset directory not found: {safe_directory}"
        )

    if not risky_directory.is_dir():
        raise FileNotFoundError(
            f"Risky dataset directory not found: {risky_directory}"
        )

    samples: list[DatasetSample] = []

    samples.extend(
        load_directory(
            safe_directory,
            label=0,
        )
    )

    samples.extend(
        load_directory(
            risky_directory,
            label=1,
        )
    )

    return Dataset(samples=samples)


def main() -> None:
    dataset = load_dataset()

    print("SentryLite Dataset")
    print("=" * 40)
    print(f"Total samples: {len(dataset.samples)}")
    print(f"Safe:          {dataset.safe_count}")
    print(f"Risky:         {dataset.risky_count}")

    if dataset.samples:
        feature_count = len(dataset.samples[0].features)

        print(f"Features:      {feature_count}")

        print()
        print("Example:")
        print(f"File:   {dataset.samples[0].path}")
        print(f"Label:  {dataset.samples[0].label}")
        print(f"Vector: {dataset.samples[0].features}")


if __name__ == "__main__":
    main()