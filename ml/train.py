"""
Train the plant identification classifier (Member 2 / Plant AI).

Approach: transfer learning via frozen feature extraction, not full
fine-tuning. A pretrained MobileNetV3-Small (ImageNet) backbone produces a
576-d embedding per image; a Logistic Regression classifier is trained on
top of those embeddings. This is deliberately simple and robust for a
small curated dataset (~18-20 images per class) - fine-tuning a full CNN
on that little data would overfit badly.

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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = pathlib.Path(__file__).parent
DATASET_DIR = HERE / "dataset"
MODEL_DIR = HERE / "model"

device = torch.device("cpu")


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

    X, y, classes = [], [], []
    for plant_dir in plant_dirs:
        plant_id = plant_dir.name
        classes.append(plant_id)
        image_files = sorted(plant_dir.glob("*.jpg"))
        print(f"[{plant_id}] embedding {len(image_files)} images...")
        for image_file in image_files:
            try:
                image = Image.open(image_file)
                features = embed_image(model, preprocess, image)
            except Exception as exc:
                print(f"  skipping {image_file.name}: {exc}")
                continue
            X.append(features)
            y.append(plant_id)

    X = np.stack(X)
    y = np.array(y)
    print(f"\nTotal samples: {len(y)}, feature dim: {X.shape[1]}, classes: {len(classes)}")

    # Heavier regularization (small C) suits this dataset: ~18-20 images per
    # class on a 576-d embedding easily overfits at the sklearn default C=1.0.
    # C=0.3 was chosen via 4-fold cross-validation (see ml/README.md).
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.3),
    )

    # Stratified + shuffled folds: plain cv=4 would use contiguous, non-shuffled
    # folds here since images are appended in class order, silently mixing
    # classes unevenly across folds and producing a meaningless accuracy number.
    try:
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf)
        print(f"4-fold stratified CV accuracy: mean={scores.mean():.3f} scores={np.round(scores, 3)}")
    except Exception as exc:
        print(f"Cross-validation skipped: {exc}")

    clf.fit(X, y)

    # Out-of-distribution guard: a softmax classifier can still be
    # confidently wrong on inputs unlike anything it was trained on (e.g. a
    # non-plant photo, or random noise - verified this happens in practice).
    # We additionally require the query embedding to be reasonably close, by
    # cosine similarity, to at least one training embedding. L2-normalize
    # first so cosine similarity is a plain dot product at inference time.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized_embeddings = X / norms

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")
    (MODEL_DIR / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    np.save(MODEL_DIR / "train_embeddings.npy", normalized_embeddings)
    print(f"\nSaved classifier to {MODEL_DIR / 'classifier.joblib'}")
    print(f"Saved classes to {MODEL_DIR / 'classes.json'}")
    print(f"Saved {len(normalized_embeddings)} reference embeddings for novelty detection")


if __name__ == "__main__":
    main()
