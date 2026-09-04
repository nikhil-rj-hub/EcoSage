"""
One-off cleanup pass (Member 2) for the expanded dataset.

The broader multi-query fetch (scientific name + "leaf"/"tree"/"<common
name> plant") pulled in a lot more images per class, but also more noise:
spot-checking found a landscape photo with a distant, barely-visible tree
and dogs in the foreground, and a temple-architecture photo where the tree
is mostly hidden behind ironwork - both technically returned by an
on-topic search query, but useless (or actively harmful) as training
images.

Manually reviewing every new image isn't feasible at this dataset size, so
this script auto-flags likely-bad images using the embeddings themselves:
each plant's *originally* fetched images (already manually spot-checked
earlier) are treated as a trusted reference; every image is scored by
cosine similarity to its class's trusted-reference centroid, and anything
below a cutoff is flagged as an outlier for manual review rather than
silently deleted.

Usage:
    python filter_dataset.py            # report only
    python filter_dataset.py --apply    # move flagged images to dataset_rejected/
"""

import argparse
import pathlib
import shutil
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchvision
from PIL import Image

HERE = pathlib.Path(__file__).parent
DATASET_DIR = HERE / "dataset"
REJECTED_DIR = HERE / "dataset_rejected"

# Number of images each class had *before* the broad multi-query expansion
# (i.e. the ones already manually spot-checked). Used as the trusted
# reference set to score everything else against.
ORIGINAL_COUNTS = {
    "plant_01": 18,
    "plant_02": 20,
    "plant_03": 19,
    "plant_04": 18,
    "plant_05": 20,
    "plant_06": 19,  # one bad image (person eating a mango) already removed
    "plant_07": 20,
    "plant_08": 20,
}

SIMILARITY_CUTOFF = 0.45


def build_feature_extractor():
    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()


def embed(model, preprocess, path: pathlib.Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    vec = features.squeeze(0).numpy()
    return vec / np.linalg.norm(vec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Move flagged images out of dataset/")
    args = parser.parse_args()

    model, preprocess = build_feature_extractor()

    total_flagged = 0
    for plant_dir in sorted(DATASET_DIR.iterdir()):
        if not plant_dir.is_dir():
            continue
        plant_id = plant_dir.name
        original_count = ORIGINAL_COUNTS.get(plant_id, 0)
        image_files = sorted(plant_dir.glob("*.jpg"), key=lambda p: int(p.stem))

        trusted_files = [f for f in image_files if int(f.stem) < original_count]
        if not trusted_files:
            print(f"[{plant_id}] no trusted reference images, skipping filter")
            continue

        trusted_embeddings = np.stack([embed(model, preprocess, f) for f in trusted_files])
        centroid = trusted_embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid)

        flagged = []
        for f in image_files:
            vec = embed(model, preprocess, f)
            similarity = float(vec @ centroid)
            if similarity < SIMILARITY_CUTOFF:
                flagged.append((f, similarity))

        print(f"[{plant_id}] {len(image_files)} images, "
              f"{len(trusted_files)} trusted reference, "
              f"{len(flagged)} flagged below similarity {SIMILARITY_CUTOFF}")
        for f, sim in flagged:
            print(f"    flagged: {f.name} (similarity={sim:.3f})")
        total_flagged += len(flagged)

        if args.apply and flagged:
            dest_dir = REJECTED_DIR / plant_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f, _ in flagged:
                shutil.move(str(f), str(dest_dir / f.name))

    print(f"\nTotal flagged: {total_flagged}")
    if not args.apply:
        print("Dry run only - rerun with --apply to move flagged images to dataset_rejected/")


if __name__ == "__main__":
    main()
