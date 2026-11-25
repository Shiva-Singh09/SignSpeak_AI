
from pathlib import Path
import hashlib
import csv
from collections import defaultdict

import cv2
import numpy as np


# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "external"
    / "RealSign"
    / "extracted"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

SPLITS = ["Training", "Validation", "Testing"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Maximum perceptual-hash Hamming distance.
# Smaller = stricter similarity detection.
PHASH_THRESHOLD = 6


# ============================================================
# IMAGE HASH FUNCTIONS
# ============================================================

def calculate_sha256(image_path):
    """Calculate exact file hash."""
    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_phash(image_path):
    """
    Calculate a 64-bit perceptual hash.

    Similar-looking images should have similar hashes,
    even if their file sizes or compression differ.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))

    resized = np.float32(resized)
    dct = cv2.dct(resized)

    low_frequency = dct[:8, :8].flatten()

    # Exclude DC coefficient while calculating median.
    median = np.median(low_frequency[1:])

    bits = low_frequency > median

    hash_value = 0

    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)

    return hash_value


def hamming_distance(hash1, hash2):
    """Count differing bits between two perceptual hashes."""
    return (hash1 ^ hash2).bit_count()


# ============================================================
# BK-TREE FOR EFFICIENT SIMILARITY SEARCH
# ============================================================

class BKTree:

    def __init__(self):
        self.root = None

    def add(self, hash_value, image_index):

        if self.root is None:
            self.root = {
                "hash": hash_value,
                "indices": [image_index],
                "children": {}
            }
            return

        node = self.root

        while True:

            distance = hamming_distance(
                hash_value,
                node["hash"]
            )

            if distance == 0:
                node["indices"].append(image_index)
                return

            if distance not in node["children"]:
                node["children"][distance] = {
                    "hash": hash_value,
                    "indices": [image_index],
                    "children": {}
                }
                return

            node = node["children"][distance]

    def search(self, hash_value, threshold):

        matches = []

        if self.root is None:
            return matches

        stack = [self.root]

        while stack:

            node = stack.pop()

            distance = hamming_distance(
                hash_value,
                node["hash"]
            )

            if distance <= threshold:
                for index in node["indices"]:
                    matches.append((index, distance))

            lower = distance - threshold
            upper = distance + threshold

            for edge_distance, child in node["children"].items():

                if lower <= edge_distance <= upper:
                    stack.append(child)

        return matches


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_images():

    records = []

    for split in SPLITS:

        split_dir = DATASET_ROOT / split

        if not split_dir.exists():
            print(f"WARNING: Missing split folder: {split_dir}")
            continue

        for class_dir in sorted(split_dir.iterdir()):

            if not class_dir.is_dir():
                continue

            label = class_dir.name

            for image_path in sorted(class_dir.iterdir()):

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                records.append({
                    "split": split,
                    "label": label,
                    "path": image_path,
                    "relative_path": str(
                        image_path.relative_to(PROJECT_ROOT)
                    )
                })

    return records


# ============================================================
# EXACT DUPLICATE AUDIT
# ============================================================

def find_exact_duplicates(records):

    hash_groups = defaultdict(list)
    unreadable = []

    print("\nCalculating SHA-256 hashes...")

    for index, record in enumerate(records):

        try:
            file_hash = calculate_sha256(record["path"])
            hash_groups[file_hash].append(index)

        except OSError as error:
            unreadable.append({
                "path": record["relative_path"],
                "error": str(error)
            })

    duplicate_groups = [
        indices
        for indices in hash_groups.values()
        if len(indices) > 1
    ]

    return duplicate_groups, unreadable


# ============================================================
# NEAR-DUPLICATE AUDIT
# ============================================================

def find_near_duplicates(records):

    tree = BKTree()
    hashes = []
    pairs = []
    invalid_images = []

    print("\nCalculating perceptual hashes...")
    print("Searching for near-duplicate images...")

    for index, record in enumerate(records):

        phash = calculate_phash(record["path"])

        if phash is None:
            invalid_images.append(record["relative_path"])
            hashes.append(None)
            continue

        hashes.append(phash)

        matches = tree.search(
            phash,
            PHASH_THRESHOLD
        )

        for previous_index, distance in matches:

            pairs.append({
                "image_1_index": previous_index,
                "image_2_index": index,
                "hamming_distance": distance
            })

        tree.add(phash, index)

        if (index + 1) % 2000 == 0:
            print(
                f"Processed {index + 1}/{len(records)} images"
            )

    return pairs, invalid_images


# ============================================================
# CSV REPORTS
# ============================================================

def write_exact_report(records, duplicate_groups):

    output_path = OUTPUT_DIR / "exact_duplicates.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "duplicate_group",
            "split",
            "label",
            "image_path"
        ])

        for group_number, indices in enumerate(
            duplicate_groups,
            start=1
        ):

            for index in indices:

                record = records[index]

                writer.writerow([
                    group_number,
                    record["split"],
                    record["label"],
                    record["relative_path"]
                ])

    return output_path


def write_near_report(records, pairs):

    output_path = OUTPUT_DIR / "near_duplicates.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "split_1",
            "label_1",
            "image_1",
            "split_2",
            "label_2",
            "image_2",
            "hamming_distance",
            "cross_split",
            "cross_class"
        ])

        for pair in pairs:

            first = records[pair["image_1_index"]]
            second = records[pair["image_2_index"]]

            writer.writerow([
                first["split"],
                first["label"],
                first["relative_path"],
                second["split"],
                second["label"],
                second["relative_path"],
                pair["hamming_distance"],
                first["split"] != second["split"],
                first["label"] != second["label"]
            ])

    return output_path


def write_unreadable_report(unreadable, invalid_images):

    output_path = OUTPUT_DIR / "duplicate_audit_unreadable.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["image_path", "reason"])

        for item in unreadable:
            writer.writerow([
                item["path"],
                item["error"]
            ])

        for path in invalid_images:
            writer.writerow([
                path,
                "OpenCV could not read image"
            ])

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SIGNSPEAK AI - DATASET DUPLICATE AUDIT")
    print("=" * 65)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}"
        )

    records = discover_images()

    print(f"\nTotal discovered images: {len(records)}")

    if not records:
        print("No images found. Check the dataset path.")
        return

    # Exact duplicate audit
    exact_groups, unreadable = find_exact_duplicates(records)

    # Near duplicate audit
    near_pairs, invalid_images = find_near_duplicates(records)

    # Save reports
    exact_file = write_exact_report(
        records,
        exact_groups
    )

    near_file = write_near_report(
        records,
        near_pairs
    )

    unreadable_file = write_unreadable_report(
        unreadable,
        invalid_images
    )

    # Summary
    cross_split_pairs = [
        pair for pair in near_pairs
        if records[pair["image_1_index"]]["split"]
        != records[pair["image_2_index"]]["split"]
    ]

    cross_class_pairs = [
        pair for pair in near_pairs
        if records[pair["image_1_index"]]["label"]
        != records[pair["image_2_index"]]["label"]
    ]

    exact_duplicate_images = sum(
        len(group) for group in exact_groups
    )

    summary_path = OUTPUT_DIR / "duplicate_audit_summary.txt"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("SIGNSPEAK AI - DUPLICATE AUDIT SUMMARY\n")
        file.write("=" * 50 + "\n")
        file.write(f"Total images: {len(records)}\n")
        file.write(f"Exact duplicate groups: {len(exact_groups)}\n")
        file.write(
            f"Images in exact duplicate groups: "
            f"{exact_duplicate_images}\n"
        )
        file.write(f"Near-duplicate pairs: {len(near_pairs)}\n")
        file.write(
            f"Near-duplicate cross-split pairs: "
            f"{len(cross_split_pairs)}\n"
        )
        file.write(
            f"Near-duplicate cross-class pairs: "
            f"{len(cross_class_pairs)}\n"
        )
        file.write(
            f"Unreadable images: "
            f"{len(unreadable) + len(invalid_images)}\n"
        )
        file.write(
            f"Perceptual hash threshold: "
            f"{PHASH_THRESHOLD}\n"
        )

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)

    print(f"Total images: {len(records)}")
    print(f"Exact duplicate groups: {len(exact_groups)}")
    print(f"Images in exact duplicate groups: {exact_duplicate_images}")
    print(f"Near-duplicate pairs: {len(near_pairs)}")
    print(f"Cross-split near-duplicate pairs: {len(cross_split_pairs)}")
    print(f"Cross-class near-duplicate pairs: {len(cross_class_pairs)}")
    print(f"Unreadable images: {len(unreadable) + len(invalid_images)}")

    print("\nReports saved:")
    print(exact_file)
    print(near_file)
    print(unreadable_file)
    print(summary_path)


if __name__ == "__main__":
    main()
