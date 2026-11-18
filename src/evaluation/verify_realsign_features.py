import csv
import math
from collections import Counter
from pathlib import Path

ROOT = Path("dataset/processed/realsign_features")
SPLITS = ("training", "validation", "testing")
EXPECTED_CLASSES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Match the exact order produced by prepare_realsign_features.py
expected_columns = ["label", "image_name"]

for hand in range(2):
    expected_columns.extend([
        f"hand_{hand}_wrist_{axis}"
        for axis in ("x", "y", "z")
    ])

    for point in range(1, 21):
        expected_columns.extend([
            f"hand_{hand}_landmark_{point:02d}_rel_{axis}"
            for axis in ("x", "y", "z")
        ])

expected_columns.extend([
    "hand_0_present",
    "hand_1_present",
    "hand_0_is_left",
    "hand_0_is_right",
    "hand_1_is_left",
    "hand_1_is_right",
])

assert len(expected_columns) == 134

print("REALSIGN FEATURE DATASET VERIFICATION")
print("=" * 55)

for split in SPLITS:
    path = ROOT / f"{split}_features.csv"
    rows = 0
    classes = Counter()
    errors = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != expected_columns:
            actual = reader.fieldnames or []
            print(f"\nExpected columns: {len(expected_columns)}")
            print(f"Actual columns: {len(actual)}")

            differences = [
                (i, expected_columns[i], actual[i] if i < len(actual) else "<missing>")
                for i in range(max(len(expected_columns), len(actual)))
                if i >= len(actual) or i >= len(expected_columns)
                or expected_columns[i] != actual[i]
            ]

            print("First column differences:", differences[:5])
            raise ValueError(f"{split}: CSV header mismatch")

        feature_columns = expected_columns[2:]

        for row_number, row in enumerate(reader, start=2):
            rows += 1
            label = row["label"]
            classes[label] += 1

            try:
                values = [float(row[c]) for c in feature_columns]

                if not all(math.isfinite(v) for v in values):
                    errors.append(f"Row {row_number}: non-finite value")
                    continue

                p0, p1 = values[126:128]
                h0_left, h0_right, h1_left, h1_right = values[128:132]

                if p0 not in (0, 1) or p1 not in (0, 1):
                    errors.append(f"Row {row_number}: invalid presence flag")

                if h0_left + h0_right != p0:
                    errors.append(f"Row {row_number}: hand 0 handedness mismatch")

                if h1_left + h1_right != p1:
                    errors.append(f"Row {row_number}: hand 1 handedness mismatch")

                for i in list(range(3, 63)) + list(range(66, 126)):
                    if abs(values[i]) > 1.00001:
                        errors.append(f"Row {row_number}: relative coordinate out of range")
                        break

                if p0 == 0 and any(values[:63]):
                    errors.append(f"Row {row_number}: absent hand 0 has features")

                if p1 == 0 and any(values[63:126]):
                    errors.append(f"Row {row_number}: absent hand 1 has features")

            except (ValueError, TypeError) as error:
                errors.append(f"Row {row_number}: {error}")

    class_pass = set(classes) == EXPECTED_CLASSES
    data_pass = not errors

    print(f"\n{split.upper()}")
    print(f"Rows: {rows}")
    print(f"Columns: {len(expected_columns)}")
    print(f"Classes: {len(classes)}")
    print(f"Class coverage: {'PASS' if class_pass else 'FAIL'}")
    print(f"Numeric and feature checks: {'PASS' if data_pass else 'FAIL'}")

    if errors:
        for error in errors[:10]:
            print("ERROR:", error)
        print(f"Total errors: {len(errors)}")

    if not class_pass or not data_pass:
        raise SystemExit(f"Verification failed for {split}")

print("\nSUCCESS: All three feature datasets passed verification.")
