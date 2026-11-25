import csv
from pathlib import Path
from collections import defaultdict

root = Path("dataset/processed/realsign_landmarks")
splits = ["Training", "Validation", "Testing"]
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

report = []

for split in splits:
    file = root / f"{split.lower()}_landmarks.csv"

    if not file.exists():
        raise FileNotFoundError(file)

    stats = defaultdict(lambda: {
        "total": 0,
        "zero": 0,
        "one": 0,
        "two": 0,
        "unreadable": 0
    })

    with file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            letter = row["label"]
            item = stats[letter]
            item["total"] += 1

            status = row["detection_status"]
            count = int(row["detected_hands"])

            if status == "unreadable":
                item["unreadable"] += 1
            elif count == 0:
                item["zero"] += 1
            elif count == 1:
                item["one"] += 1
            elif count == 2:
                item["two"] += 1

    print(f"\n{'=' * 76}")
    print(f"{split.upper()} DETECTION QUALITY")
    print("=" * 76)
    print(
        f"{'Letter':<8}{'Total':>8}{'0 hands':>10}"
        f"{'1 hand':>10}{'2 hands':>10}"
        f"{'Unreadable':>13}{'Detected %':>13}"
    )

    for letter in letters:
        s = stats[letter]
        valid = s["total"] - s["unreadable"]
        detected = s["one"] + s["two"]
        rate = (detected / valid * 100) if valid else 0

        print(
            f"{letter:<8}{s['total']:>8}{s['zero']:>10}"
            f"{s['one']:>10}{s['two']:>10}"
            f"{s['unreadable']:>13}{rate:>12.1f}%"
        )

        report.append({
            "split": split,
            "letter": letter,
            **s,
            "detected": detected,
            "detection_rate_percent": round(rate, 2)
        })

    print("-" * 76)

    total = sum(s["total"] for s in stats.values())
    zero = sum(s["zero"] for s in stats.values())
    one = sum(s["one"] for s in stats.values())
    two = sum(s["two"] for s in stats.values())
    unreadable = sum(s["unreadable"] for s in stats.values())

    valid = total - unreadable
    detected = one + two

    print("Total images:", total)
    print("Detected hands:", detected)
    print("No hands:", zero)
    print("Unreadable:", unreadable)
    print(
        "Detection rate:",
        f"{detected / valid * 100:.2f}%" if valid else "N/A"
    )

report_file = root / "detection_quality_by_letter.csv"

with report_file.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=report[0].keys())
    writer.writeheader()
    writer.writerows(report)

print("\nQuality report saved to:", report_file)
