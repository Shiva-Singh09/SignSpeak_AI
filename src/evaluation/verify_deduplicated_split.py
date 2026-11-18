
import csv
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "realsign_deduplicated"
)

IMAGE_ROOT = ROOT / "images"
MANIFEST = ROOT / "reports" / "dataset_manifest.csv"

def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():

    print("=" * 65)
    print("VERIFYING DEDUPLICATED REALSIGN DATASET")
    print("=" * 65)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    manifest_rows = []

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    print(f"\nManifest records: {len(manifest_rows)}")

    split_counts = Counter()
    class_counts = defaultdict(Counter)
    hash_splits = defaultdict(set)
    missing_files = []
    hash_mismatches = []

    for row in manifest_rows:

        split = row["new_split"]
        label = row["label"]
        path = Path(row["destination_path"])

        split_counts[split] += 1
        class_counts[split][label] += 1

        if not path.exists():
            missing_files.append(str(path))
            continue

        actual_hash = sha256(path)

        if actual_hash != row["sha256"]:
            hash_mismatches.append(str(path))

        hash_splits[actual_hash].add(split)

    print("\n[1] Split counts")

    for split in ("Training", "Validation", "Testing"):
        print(f"{split}: {split_counts[split]}")

    print("\n[2] Class coverage")

    for split in ("Training", "Validation", "Testing"):
        classes = class_counts[split]
        print(f"{split}: {len(classes)} classes")

        missing = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set(classes))

        if missing:
            print("Missing classes:", missing)

    print("\n[3] File integrity")
    print("Missing files:", len(missing_files))
    print("Hash mismatches:", len(hash_mismatches))

    leaking_hashes = {
        h: splits for h, splits in hash_splits.items()
        if len(splits) > 1
    }

    print("\n[4] Exact duplicate leakage")
    print("Hashes appearing in multiple splits:", len(leaking_hashes))

    print("\n[5] Final result")

    if (
        len(manifest_rows) == 25975
        and not missing_files
        and not hash_mismatches
        and not leaking_hashes
        and all(
            len(class_counts[s]) == 26
            for s in ("Training", "Validation", "Testing")
        )
    ):
        print("PASS: Dataset integrity verified.")
    else:
        print("FAIL: Inspect the reported issues.")

if __name__ == "__main__":
    main()
