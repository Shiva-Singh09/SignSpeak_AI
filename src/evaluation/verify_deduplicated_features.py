
import csv
from pathlib import Path

import numpy as np


BASE = Path("dataset/processed/realsign_deduplicated/features")

EXPECTED_ROWS = {
    "training": 18183,
    "validation": 2597,
    "testing": 5195,
}

EXPECTED_CLASSES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

FEATURE_COUNT = 132
EXPECTED_COLUMNS = 134

SPLITS = ("training", "validation", "testing")


def verify_split(split):
    path = BASE / f"{split}_features.csv"

    print(f"\n{split.upper()}")
    print("-" * 55)

    if not path.exists():
        raise FileNotFoundError(path)

    row_count = 0
    class_counts = {}

    invalid_numbers = 0
    invalid_flags = 0
    missing_hand_errors = 0
    handedness_errors = 0
    incorrect_columns = 0

    with path.open(
        "r", encoding="utf-8", newline=""
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV header is missing")

        if len(reader.fieldnames) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Expected {EXPECTED_COLUMNS} columns, "
                f"got {len(reader.fieldnames)}"
            )

        feature_names = reader.fieldnames[2:]

        if len(feature_names) != FEATURE_COUNT:
            raise ValueError("Feature count mismatch")

        for row in reader:
            row_count += 1

            label = row["label"]
            class_counts[label] = (
                class_counts.get(label, 0) + 1
            )

            try:
                values = np.array([
                    float(row[name])
                    for name in feature_names
                ], dtype=np.float32)

            except (ValueError, TypeError):
                invalid_numbers += 1
                continue

            if not np.isfinite(values).all():
                invalid_numbers += 1
                continue

            # Feature positions:
            # 0-62   : hand_0 features
            # 63-125 : hand_1 features
            # 126    : hand_0_present
            # 127    : hand_1_present
            # 128    : hand_0_is_left
            # 129    : hand_0_is_right
            # 130    : hand_1_is_left
            # 131    : hand_1_is_right

            flags = values[126:132]

            if not np.isin(flags, [0, 1]).all():
                invalid_flags += 1
                continue

            present_0, present_1 = flags[0:2]
            left_0, right_0 = flags[2:4]
            left_1, right_1 = flags[4:6]

            # Presence must agree with handedness flags.
            if present_0 != left_0 + right_0:
                handedness_errors += 1

            if present_1 != left_1 + right_1:
                handedness_errors += 1

            # Missing-hand coordinate features must be zero.
            if present_0 == 0:
                if not np.allclose(
                    values[0:63], 0, atol=1e-7
                ):
                    missing_hand_errors += 1

            if present_1 == 0:
                if not np.allclose(
                    values[63:126], 0, atol=1e-7
                ):
                    missing_hand_errors += 1

    missing_classes = EXPECTED_CLASSES - set(class_counts)
    unexpected_classes = set(class_counts) - EXPECTED_CLASSES

    row_error = row_count != EXPECTED_ROWS[split]

    print(f"Rows: {row_count}")
    print(f"Expected rows: {EXPECTED_ROWS[split]}")
    print(f"Columns: {EXPECTED_COLUMNS}")
    print(f"Features per image: {FEATURE_COUNT}")
    print(f"Classes: {len(class_counts)}")
    print(f"Invalid numeric values: {invalid_numbers}")
    print(f"Invalid binary flags: {invalid_flags}")
    print(f"Missing-hand coordinate errors: {missing_hand_errors}")
    print(f"Handedness flag errors: {handedness_errors}")
    print(f"Missing classes: {sorted(missing_classes)}")
    print(f"Unexpected classes: {sorted(unexpected_classes)}")

    passed = (
        not row_error
        and len(class_counts) == 26
        and not missing_classes
        and not unexpected_classes
        and invalid_numbers == 0
        and invalid_flags == 0
        and missing_hand_errors == 0
        and handedness_errors == 0
    )

    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return passed


def main():
    print("=" * 60)
    print("VERIFYING DEDUPLICATED FEATURE FILES")
    print("=" * 60)

    results = []

    for split in SPLITS:
        results.append(verify_split(split))

    print("\n" + "=" * 60)

    if all(results):
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")

    print("=" * 60)


if __name__ == "__main__":
    main()
