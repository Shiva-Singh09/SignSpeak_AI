
import csv
from pathlib import Path
from collections import Counter, defaultdict

FEATURE_FILE = Path(
    "dataset/processed/realsign_deduplicated/features/validation_features.csv"
)

REPORT_FILE = Path(
    "outputs/reports/deduplicated_d_audit/hand_presence_check.txt"
)


def main():

    print("=" * 65)
    print("D FALSE POSITIVE - HAND PRESENCE INVESTIGATION")
    print("=" * 65)

    groups = defaultdict(Counter)
    class_counts = defaultdict(Counter)

    with FEATURE_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames

        print("\nFEATURE CSV HAND FLAGS")
        print("-" * 45)

        print("Last six feature columns:")
        print(fieldnames[-6:])

        for row in reader:

            actual = row["label"]

            # Feature order:
            # 126 coordinates,
            # hand_0_present, hand_1_present,
            # four handedness flags.
            values = [
                float(row[name])
                for name in fieldnames[2:]
            ]

            hand0 = int(values[126])
            hand1 = int(values[127])

            if hand0 and hand1:
                presence = "TWO_HANDS"
            elif hand0:
                presence = "ONE_HAND"
            elif hand1:
                presence = "ONE_HAND"
            else:
                presence = "NO_HANDS"

            class_counts[actual][presence] += 1

            # All no-hand feature vectors should be zero.
            if presence == "NO_HANDS":
                if any(value != 0 for value in values):
                    raise ValueError(
                        f"Nonzero features in no-hand sample: "
                        f"{row['image_name']}"
                    )

    print("\nHAND PRESENCE BY ACTUAL CLASS")
    print("-" * 65)

    total = Counter()

    for label in sorted(class_counts):

        counts = class_counts[label]

        print(
            f"{label}: "
            f"no hands={counts['NO_HANDS']}, "
            f"one hand={counts['ONE_HAND']}, "
            f"two hands={counts['TWO_HANDS']}"
        )

        total.update(counts)

    print("\nOVERALL VALIDATION HAND PRESENCE")
    print("-" * 45)

    for key in ["NO_HANDS", "ONE_HAND", "TWO_HANDS"]:
        print(f"{key}: {total[key]}")

    print("\n" + "=" * 65)
    print("CHECK COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
