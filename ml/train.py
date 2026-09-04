"""
Train the plant identification classifier (Member 2 / Plant AI).

Approach: transfer learning via frozen feature extraction, not full
fine-tuning. A pretrained MobileNetV3-Small (ImageNet) backbone produces a
576-d embedding per image; a Logistic Regression classifier is trained on
top of those embeddings. This is deliberately simple and robust for a
small curated dataset - fine-tuning a full CNN on this little data would
overfit badly.

Each source photo is also embedded under several light augmentations
(crop/flip/color-jitter/rotation) to multiply the effective training set
without needing more raw photos. Evaluation uses StratifiedGroupKFold so
augmented copies of the same source photo can't leak across train/validation
folds - see the CV code below for why that matters.

Usage:
    python train.py

Reads:  ml/dataset/<plant_id>/*.jpg
Writes: ml/model/classifier.joblib   (sklearn LogisticRegression)
        ml/model/classes.json        (index -> plant_id mapping)
"""

import json
import pathlib

import joblib
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = pathlib.Path(__file__).parent
DATASET_DIR = HERE / "dataset"
MODEL_DIR = HERE / "model"

device = torch.device("cpu")

# Light augmentation to multiply effective training samples per image: the
# frozen backbone isn't perfectly invariant to these, so each augmented
# view produces a genuinely different embedding, giving the classifier more
# to generalize from without needing more raw photos.
AUGMENTATIONS_PER_IMAGE = 4
AUGMENT = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
    T.RandomRotation(degrees=12),
])


def build_feature_extractor():
    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()  # expose the 576-d pooled features
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def embed_image(model, preprocess, image: Image.Image) -> np.ndarray:
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze(0).numpy()


def main() -> None:
    model, preprocess = build_feature_extractor()

    plant_dirs = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())
    if not plant_dirs:
        raise SystemExit(f"No plant folders found under {DATASET_DIR}")

    # `groups` ties every augmented view back to its source photo. This
    # matters for honest evaluation: if augmented copies of the same photo
    # could land in both the train and validation fold, cross-validation
    # would leak and report inflated, meaningless accuracy. StratifiedGroupKFold
    # below keeps all views of one source photo in the same fold.
    X, y, groups, classes = [], [], [], []
    original_embeddings_by_class = {}  # non-augmented only - see novelty guard below
    group_id = 0
    for plant_dir in plant_dirs:
        plant_id = plant_dir.name
        classes.append(plant_id)
        image_files = sorted(plant_dir.glob("*.jpg"))
        print(f"[{plant_id}] embedding {len(image_files)} images "
              f"(x{1 + AUGMENTATIONS_PER_IMAGE} incl. augmentations)...")
        originals = []
        for image_file in image_files:
            try:
                image = Image.open(image_file).convert("RGB")
            except Exception as exc:
                print(f"  skipping {image_file.name}: {exc}")
                continue

            # Original, un-augmented view.
            original_embedding = embed_image(model, preprocess, image)
            X.append(original_embedding)
            y.append(plant_id)
            groups.append(group_id)
            originals.append(original_embedding / np.linalg.norm(original_embedding))

            for _ in range(AUGMENTATIONS_PER_IMAGE):
                augmented = AUGMENT(image)
                X.append(embed_image(model, preprocess, augmented))
                y.append(plant_id)
                groups.append(group_id)

            group_id += 1
        original_embeddings_by_class[plant_id] = np.stack(originals)

    X = np.stack(X)
    y = np.array(y)
    groups = np.array(groups)
    print(f"\nTotal samples: {len(y)} ({group_id} source photos), "
          f"feature dim: {X.shape[1]}, classes: {len(classes)}")

    # Heavier regularization (small C) suits this dataset: even with
    # augmentation, ~20-100 source photos per class on a 576-d embedding
    # overfits easily at the sklearn default C=1.0.
    # C=0.3 was chosen via cross-validation (see ml/README.md).
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.3),
    )

    # StratifiedGroupKFold: keeps class balance across folds (Stratified)
    # AND keeps every augmented view of one source photo in the same fold
    # (Group), so no photo's augmented siblings leak between train/val.
    try:
        # Base the fold count on source photos per class, not augmented
        # sample count, since that's the real constraint for grouped folds.
        photos_per_class = {}
        for plant_id, gid in zip(y, groups):
            photos_per_class.setdefault(plant_id, set()).add(gid)
        min_photos_per_class = min(len(v) for v in photos_per_class.values())
        n_splits = max(2, min(4, min_photos_per_class))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=sgkf, groups=groups)
        print(f"{n_splits}-fold grouped/stratified CV accuracy: "
              f"mean={scores.mean():.3f} scores={np.round(scores, 3)}")
    except Exception as exc:
        print(f"Cross-validation skipped: {exc}")

    clf.fit(X, y)

    # Out-of-distribution guard, v2.
    #
    # v1 (checking similarity to ANY of the training embeddings, pooled
    # across all classes) worked on the small initial dataset but was
    # verified to break down as the dataset grew: with thousands of diverse
    # augmented embeddings across 8 classes, almost any photo - including
    # confirmed non-project species like rose, hibiscus, and sunflower -
    # scores high similarity to *something* in the pool purely by chance.
    # Measured result before this fix: 8 of 10 test photos of unsupported
    # species were confidently (0.86-1.0) misidentified as one of our 8.
    #
    # Fix: instead of "similar to anything we've seen," require the query to
    # be similar to *photos of the specific class the classifier predicted*,
    # using a per-class threshold calibrated from that class's own data
    # (the 5th percentile of how similar the class's own photos are to each
    # other) rather than one global cutoff. Uses only original, non-augmented
    # embeddings for this - augmented copies of the same source photo are
    # near-duplicates of each other and would make same-class similarity
    # look artificially high, undermining the calibration.
    #
    # Verified after this fix: 9 of 10 unsupported-species test photos are
    # correctly rejected (see ml/README.md for the one that still slips
    # through, and why this can't be made perfect with this approach).
    class_similarity_thresholds = {}
    for plant_id, embeddings in original_embeddings_by_class.items():
        if len(embeddings) < 2:
            class_similarity_thresholds[plant_id] = 0.0
            continue
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, -1)
        same_class_nn_sim = sim_matrix.max(axis=1)
        class_similarity_thresholds[plant_id] = float(np.percentile(same_class_nn_sim, 5))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")
    (MODEL_DIR / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    np.savez(
        MODEL_DIR / "class_reference_embeddings.npz",
        **original_embeddings_by_class,
    )
    (MODEL_DIR / "class_similarity_thresholds.json").write_text(
        json.dumps(class_similarity_thresholds, indent=2), encoding="utf-8"
    )
    print(f"\nSaved classifier to {MODEL_DIR / 'classifier.joblib'}")
    print(f"Saved classes to {MODEL_DIR / 'classes.json'}")
    print("Per-class novelty-guard thresholds (5th percentile same-class similarity):")
    for plant_id, threshold in class_similarity_thresholds.items():
        print(f"  {plant_id}: {threshold:.3f}")


if __name__ == "__main__":
    main()
