import csv
from pathlib import Path
from collections import Counter

source = Path("dataset/processed/realsign_landmarks")
destination = Path("dataset/processed/realsign_clean")

destination.mkdir(parents=True, exist_ok=True)

splits = ["Training", "Validation", "Testing"]
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

for split in splits:
    input_file = source / f"{split.lower()}_landmarks.csv"
    output_file = destination / f"{split.lower()}_landmarks.csv"

    counts = Counter()
    class_counts = Counter()

    with input_file.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)

        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {input_file}")

        with output_file.open(
            "w", encoding="utf-8", newline=""
        ) as fout:
            writer = csv.DictWriter(
                fout, fieldnames=reader.fieldnames
            )
            writer.writeheader()

            for row in reader:
                counts["original"] += 1

                if row["detection_status"] != "ok":
                    counts["excluded_unreadable"] += 1
                    continue

                hands = int(row["detected_hands"])

                if hands == 0:
                    counts["excluded_no_hand"] += 1
                    continue

                writer.writerow(row)
                counts["retained"] += 1
                counts[f"{hands}_hand"] += 1
                class_counts[row["label"]] += 1

    print(f"\n{split}")
    print("-" * 40)
    print("Original images:", counts["original"])
    print("Retained:", counts["retained"])
    print("One hand:", counts["1_hand"])
    print("Two hands:", counts["2_hand"])
    print("Excluded, no hand:", counts["excluded_no_hand"])
    print("Excluded, unreadable:", counts["excluded_unreadable"])
    print("Output:", output_file)

    missing = [
        letter for letter in letters
        if class_counts[letter] == 0
    ]
    print("Classes with no retained samples:", missing or "None")

print("\nCleaning completed.")
print("Original landmark CSV files were not modified.")
print("Cleaned dataset folder:", destination)
