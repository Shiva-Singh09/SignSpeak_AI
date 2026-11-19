
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "realsign_features"
    / "training_features.csv"
)

PAIRS = [
    ("Q", "N"),
    ("S", "Q"),
    ("W", "M"),
    ("V", "F"),
    ("Z", "B"),
]

GROUPS = {
    "Hand 0": (0, 63),
    "Hand 1": (63, 126),
    "Hand flags": (126, 132),
    "All features": (0, 132),
}


def calculate_distance(data, label_a, label_b, start, end):
    features = data.iloc[:, 2:].to_numpy(dtype=np.float64)

    mask_a = data["label"].astype(str).to_numpy() == label_a
    mask_b = data["label"].astype(str).to_numpy() == label_b

    a = features[mask_a, start:end]
    b = features[mask_b, start:end]

    if len(a) < 2 or len(b) < 2:
        return float("nan")

    mean_difference = np.mean(a, axis=0) - np.mean(b, axis=0)

    # Pooled within-class standard deviation
    variance_a = np.var(a, axis=0, ddof=1)
    variance_b = np.var(b, axis=0, ddof=1)

    pooled_variance = (
        (len(a) - 1) * variance_a
        + (len(b) - 1) * variance_b
    ) / (len(a) + len(b) - 2)

    # Ignore features with almost no variation
    valid = pooled_variance > 1e-10

    if not np.any(valid):
        return float("nan")

    standardized_difference = (
        mean_difference[valid]
        / np.sqrt(pooled_variance[valid])
    )

    return float(
        np.sqrt(np.mean(standardized_difference ** 2))
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    data = pd.read_csv(DATA_PATH)

    if data.shape[1] != 134:
        raise ValueError(
            f"Expected 134 columns, found {data.shape[1]}"
        )

    print("\nCONFUSING CLASS FEATURE COMPARISON")
    print("=" * 65)
    print(f"Training samples: {len(data)}")

    print("\nClass sample counts:")
    print(data["label"].value_counts().sort_index().to_string())

    results = []

    for label_a, label_b in PAIRS:
        print(f"\n{'-' * 65}")
        print(f"PAIR: {label_a} vs {label_b}")

        for group_name, (start, end) in GROUPS.items():
            distance = calculate_distance(
                data, label_a, label_b, start, end
            )

            results.append({
                "pair": f"{label_a}-{label_b}",
                "feature_group": group_name,
                "distance": distance,
            })

            print(f"{group_name:15s}: {distance:.4f}")

    report = pd.DataFrame(results)

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / "confusing_class_feature_distances.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
