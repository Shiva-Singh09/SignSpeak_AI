
import csv
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "dataset" / "raw" / "alphabet_landmarks.csv"

samples = defaultdict(Counter)
sample_labels = {}

with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        sample_id = row["sample_id"]
        label = row["label"]

        sample_labels[sample_id] = label
        samples[sample_id][int(row["hand_count"])] += 1

print("\n========== DATASET QUALITY REPORT ==========\n")

total_frames = 0
total_zero = 0

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    print(f"\nLETTER {letter}")
    print("-" * 35)

    letter_samples = [
        sid for sid in samples
        if sample_labels[sid] == letter
    ]

    for sid in sorted(letter_samples):
        counts = samples[sid]
        total = sum(counts.values())
        zero = counts[0]
        one = counts[1]
        two = counts[2]

        zero_percent = (zero / total) * 100 if total else 0

        total_frames += total
        total_zero += zero

        status = "OK" if zero_percent <= 20 else "CHECK"

        print(
            f"Sample {sid[:8]} | "
            f"0 hands: {zero:2d} | "
            f"1 hand: {one:2d} | "
            f"2 hands: {two:2d} | "
            f"No-hand: {zero_percent:5.1f}% | "
            f"{status}"
        )

print("\n========== OVERALL RESULT ==========")

overall = (total_zero / total_frames) * 100 if total_frames else 0

print("Total frames:", total_frames)
print("Frames with no hands:", total_zero)
print("Overall no-hand percentage:", round(overall, 2), "%")

print("\nCHECK means inspect this sample; it does not automatically mean it is invalid.")
