
import csv
import json
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE = Path("dataset/processed/realsign_deduplicated")

SOURCE_DIR = BASE / "features"

OUTPUT_DIR = BASE / "features_hand_present"

REPORT_DIR = OUTPUT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


SPLITS = [
    "training",
    "validation",
]


# ============================================================
# FILTER FUNCTION
# ============================================================

def filter_split(split):

    input_file = SOURCE_DIR / f"{split}_features.csv"

    output_file = OUTPUT_DIR / f"{split}_features.csv"

    total = 0
    retained = 0
    removed = 0

    class_stats = {}

    with input_file.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as source:

        reader = csv.DictReader(source)

        fieldnames = reader.fieldnames

        required = {
            "label",
            "image_name",
            "hand_0_present",
            "hand_1_present",
        }

        if not required.issubset(fieldnames):
            raise ValueError(
                f"Required columns missing in {input_file}"
            )

        with output_file.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as destination:

            writer = csv.DictWriter(
                destination,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for row in reader:

                total += 1

                label = row["label"]

                if label not in class_stats:

                    class_stats[label] = {
                        "total": 0,
                        "retained": 0,
                        "removed": 0,
                    }

                class_stats[label]["total"] += 1

                hand_0 = int(float(row["hand_0_present"]))
                hand_1 = int(float(row["hand_1_present"]))

                # Retain samples with at least one detected hand.
                if hand_0 == 1 or hand_1 == 1:

                    writer.writerow(row)

                    retained += 1

                    class_stats[label]["retained"] += 1

                else:

                    removed += 1

                    class_stats[label]["removed"] += 1

    print(f"\n{split.upper()}")
    print("-" * 50)

    print(f"Original samples: {total}")
    print(f"Retained samples: {retained}")
    print(f"Removed no-hand samples: {removed}")

    print("\nClass distribution:")

    for label in sorted(class_stats):

        stats = class_stats[label]

        print(
            f"{label}: "
            f"total={stats['total']}, "
            f"retained={stats['retained']}, "
            f"removed={stats['removed']}"
        )

    return {
        "original_samples": total,
        "retained_samples": retained,
        "removed_no_hand_samples": removed,
        "class_statistics": class_stats,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SIGNSPEAK - NO-HAND FILTERING EXPERIMENT")
    print("=" * 65)

    results = {}

    for split in SPLITS:

        results[split] = filter_split(split)

    report_file = REPORT_DIR / "filtering_summary.json"

    with report_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("FILTERING COMPLETED")
    print("=" * 65)

    print(f"Output directory: {OUTPUT_DIR}")

    print(f"Summary report: {report_file}")

    print("\nOriginal feature files remain unchanged.")


if __name__ == "__main__":

    main()
