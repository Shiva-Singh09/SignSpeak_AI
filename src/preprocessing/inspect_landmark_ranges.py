import csv
import numpy as np
from pathlib import Path

path = Path("dataset/processed/realsign_clean/training_landmarks.csv")

coordinate_columns = [
    f"hand_{hand}_{point:02d}_{axis}"
    for hand in range(2)
    for point in range(21)
    for axis in ("x", "y", "z")
]

values = {column: [] for column in coordinate_columns}

with path.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    missing = [
        column for column in coordinate_columns
        if column not in (reader.fieldnames or [])
    ]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    for row in reader:
        for column in coordinate_columns:
            values[column].append(float(row[column]))

print("LANDMARK COORDINATE RANGE INSPECTION")
print("-" * 55)

for hand in range(2):
    print(f"\nHand {hand}:")

    for axis in ("x", "y", "z"):
        columns = [
            f"hand_{hand}_{point:02d}_{axis}"
            for point in range(21)
        ]

        combined = np.array([
            value
            for column in columns
            for value in values[column]
            if value != 0
        ])

        if combined.size:
            print(
                f"  {axis.upper()}: "
                f"min={combined.min():.5f}, "
                f"max={combined.max():.5f}, "
                f"mean={combined.mean():.5f}, "
                f"std={combined.std():.5f}"
            )
        else:
            print(f"  {axis.upper()}: no non-zero values")

print("\nInspection completed. No dataset files were changed.")
