"""
One-off cleanup: remove exact-duplicate images from ml/dataset/.

Found via `sha256sum` scan: repeated `fetch_dataset.py` runs across
different sessions each reset their own `seen_urls` dedup set, so the same
Commons image could be (and was - 414 duplicate groups out of 1340 files)
downloaded and appended again under a new filename in a later run. Beyond
wasting space, this silently undermines train.py's StratifiedGroupKFold
leak-prevention: two files with identical content get different `group_id`s,
so their augmented copies can land in different CV folds - the exact kind
of leak grouping was meant to prevent, just via a different mechanism.

Usage:
    python dedupe_dataset.py            # report only
    python dedupe_dataset.py --apply    # actually delete duplicates
"""

import argparse
import hashlib
import pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).parent
DATASET_DIR = HERE / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    by_hash = defaultdict(list)
    for plant_dir in sorted(DATASET_DIR.iterdir()):
        if not plant_dir.is_dir():
            continue
        for f in sorted(plant_dir.glob("*.jpg")):
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            by_hash[digest].append(f)

    total_files = sum(len(v) for v in by_hash.values())
    duplicate_groups = {h: files for h, files in by_hash.items() if len(files) > 1}
    total_duplicates_to_remove = sum(len(files) - 1 for files in duplicate_groups.values())

    print(f"Total files: {total_files}")
    print(f"Duplicate content groups: {len(duplicate_groups)}")
    print(f"Files to remove (keeping one per group): {total_duplicates_to_remove}")

    removed = 0
    for files in duplicate_groups.values():
        keep, *extras = sorted(files)  # keep the lowest-numbered/first file
        for f in extras:
            print(f"  {'removing' if args.apply else 'would remove'}: {f.relative_to(DATASET_DIR)} (dup of {keep.relative_to(DATASET_DIR)})")
            if args.apply:
                f.unlink()
                removed += 1

    if args.apply:
        print(f"\nRemoved {removed} duplicate files.")
    else:
        print("\nDry run only - rerun with --apply to actually delete duplicates.")


if __name__ == "__main__":
    main()
