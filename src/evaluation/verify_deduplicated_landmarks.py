
import csv
import math
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT / "dataset" / "processed"
    / "realsign_deduplicated" / "landmarks"
)

SPLITS = ["training", "validation", "testing"]

EXPECTED_ROWS = {
    "training": 18183,
    "validation": 2597,
    "testing": 5195,
}

EXPECTED_COLUMNS = 135

def main():

    print("=" * 65)
    print("VERIFYING DEDUPLICATED LANDMARK CSV FILES")
    print("=" * 65)

    overall_pass = True

    for split in SPLITS:

        path = ROOT / f"{split}_landmarks.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        rows = 0
        invalid_numbers = 0
        invalid_hand_counts = 0
        missing_hand_nonzero = 0
        unreadable = 0
        hand_distribution = Counter()
        classes = set()

        with path.open(
            "r", newline="", encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            columns = len(reader.fieldnames)

            for row in reader:

                rows += 1
                classes.add(row["label"])

                count = int(row["detected_hands"])
                hand_distribution[count] += 1

                if row["detection_status"] == "unreadable":
                    unreadable += 1

                categories = [
                    row["hand_0_category"],
                    row["hand_1_category"],
                ]

                actual_count = sum(
                    category in ("Left", "Right")
                    for category in categories
                )

                if actual_count != count:
                    invalid_hand_counts += 1

                # Check all 126 coordinate values.
                for hand in range(2):
                    for point in range(21):
                        for axis in ("x", "y", "z"):

                            column = (
                                f"hand_{hand}_{point:02d}_{axis}"
                            )

                            value = float(row[column])

                            if not math.isfinite(value):
                                invalid_numbers += 1

                            # A missing hand must have zero coordinates.
                            if categories[hand] == "":
                                if value != 0.0:
                                    missing_hand_nonzero += 1

        split_pass = (
            rows == EXPECTED_ROWS[split]
            and columns == EXPECTED_COLUMNS
            and len(classes) == 26
            and invalid_numbers == 0
            and invalid_hand_counts == 0
            and missing_hand_nonzero == 0
            and unreadable == 0
        )

        overall_pass = overall_pass and split_pass

        print(f"\n{split.upper()}")
        print("-" * 40)
        print("Rows:", rows)
        print("Columns:", columns)
        print("Classes:", len(classes))
        print("Hand distribution:", dict(hand_distribution))
        print("Invalid numbers:", invalid_numbers)
        print("Hand-count mismatches:", invalid_hand_counts)
        print("Missing-hand nonzero coordinates:", missing_hand_nonzero)
        print("Unreadable rows:", unreadable)
        print("Result:", "PASS" if split_pass else "FAIL")

    print("\n" + "=" * 65)
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    print("=" * 65)


if __name__ == "__main__":
    main()
