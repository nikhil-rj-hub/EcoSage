"""
Test whether test-time augmentation (TTA) actually improves accuracy,
honestly, before adding it to production.

Methodology: for each held-out validation group (one source photo) in a
StratifiedGroupKFold split, we already have 5 embeddings for it (the
original + 4 augmented views, all sharing the same true label - this is
exactly what train.py already computes for training). Compare:
  - "Single view" accuracy: score using only the original (unaugmented)
    embedding, per photo - what production currently does.
  - "TTA" accuracy: average predict_proba across all 5 views for that
    photo, then take the argmax - what TTA would do at inference if we
    generated 4 augmented copies of the uploaded image ourselves.

This reuses the exact same embeddings already computed for train.py's own
cross-validation, so it's a fair, apples-to-apples comparison, not a new
guess.
"""
import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold

import train as train_mod

DATASET_DIR = pathlib.Path(__file__).parent / "dataset"


def build_dataset():
    model, preprocess = train_mod.build_feature_extractor()
    X, y, groups = [], [], []
    gid = 0
    for plant_dir in sorted(DATASET_DIR.iterdir()):
        if not plant_dir.is_dir():
            continue
        for f in sorted(plant_dir.glob("*.jpg")):
            img = Image.open(f).convert("RGB")
            X.append(train_mod.embed_image(model, preprocess, img))
            y.append(plant_dir.name)
            groups.append(gid)
            for _ in range(train_mod.AUGMENTATIONS_PER_IMAGE):
                aug = train_mod.AUGMENT(img)
                X.append(train_mod.embed_image(model, preprocess, aug))
                y.append(plant_dir.name)
                groups.append(gid)
            gid += 1
    return np.stack(X), np.array(y), np.array(groups)


def main():
    print("Embedding dataset with DINO (reusing train.py's exact pipeline)...")
    X, y, groups = build_dataset()
    print(f"Total samples: {len(y)}, groups: {groups.max()+1}")

    photos_per_class = {}
    for label, gid in zip(y, groups):
        photos_per_class.setdefault(label, set()).add(gid)
    min_photos = min(len(v) for v in photos_per_class.values())
    n_splits = max(2, min(4, min_photos))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    single_view_accs = []
    tta_accs = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256,), alpha=0.01, max_iter=800, random_state=42),
        )
        clf.fit(X[train_idx], y[train_idx])

        # Group validation rows by photo (group id); first row per group in
        # our construction order is always the original (unaugmented) view.
        val_groups = groups[val_idx]
        unique_groups = np.unique(val_groups)

        single_correct = 0
        tta_correct = 0
        for g in unique_groups:
            rows = val_idx[val_groups == g]
            # rows are in dataset order: original first, then 4 augmented.
            original_row = rows[0]
            true_label = y[original_row]

            single_pred = clf.predict(X[original_row:original_row+1])[0]
            single_correct += int(single_pred == true_label)

            probs = clf.predict_proba(X[rows])
            avg_probs = probs.mean(axis=0)
            tta_pred = clf.classes_[np.argmax(avg_probs)]
            tta_correct += int(tta_pred == true_label)

        single_acc = single_correct / len(unique_groups)
        tta_acc = tta_correct / len(unique_groups)
        single_view_accs.append(single_acc)
        tta_accs.append(tta_acc)
        print(f"Fold {fold}: single-view={single_acc:.3f}, TTA(5-view avg)={tta_acc:.3f}")

    print()
    print(f"Mean single-view accuracy: {np.mean(single_view_accs):.3f}")
    print(f"Mean TTA accuracy:         {np.mean(tta_accs):.3f}")
    print(f"Difference: {np.mean(tta_accs) - np.mean(single_view_accs):+.3f}")


if __name__ == "__main__":
    main()
