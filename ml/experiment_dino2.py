"""
Follow-up to experiment_dino.py: DINO gave a real jump (71.0% -> 80.7%
with an MLP head). Push further:
  1. Bigger/tuned MLP on DINO alone.
  2. Concatenate DINO + MobileNetV3 embeddings (different training
     paradigms - self-supervised ViT vs supervised CNN - so unlike the
     earlier MobileNet-vs-ResNet18 comparison, these may carry genuinely
     complementary signal, not just correlated noise).
"""
import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np
import torch
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

import train as train_mod  # for MobileNet + AUGMENT, to stay consistent

DATASET_DIR = pathlib.Path(__file__).parent / "dataset"

X_dino = np.load("dino_X.npy")
y = np.load("dino_y.npy", allow_pickle=True)
groups = np.load("dino_groups.npy")


def cv_accuracy(clf, X, y, groups):
    photos_per_class = {}
    for label, gid in zip(y, groups):
        photos_per_class.setdefault(label, set()).add(gid)
    min_photos = min(len(v) for v in photos_per_class.values())
    n_splits = max(2, min(4, min_photos))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=sgkf, groups=groups)
    return scores.mean(), scores


print("=== Bigger/tuned MLPs on DINO alone ===")
for hidden, alpha in [((256,), 0.01), ((256, 64), 0.01), ((128,), 0.001), ((128,), 0.1), ((64,), 0.01)]:
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=800, random_state=42),
    )
    mean, scores = cv_accuracy(clf, X_dino, y, groups)
    print(f"MLP hidden={hidden} alpha={alpha}: mean={mean:.3f} scores={np.round(scores,3)}")

print("\n=== Re-embedding with MobileNetV3 to build a matching, aligned feature set ===")
model_mb, preprocess_mb = train_mod.build_feature_extractor()

X_mb = []
gid_current = -1
img_cache = {}
# Rebuild MobileNet embeddings in the SAME order/augmentation draw isn't
# reproducible exactly (augmentation is random), so instead re-embed the
# same underlying images fresh for both, using a fixed seed per image via
# re-deriving from the dataset directly, original + augmented counts
# matching dino_groups order (grouped by plant dir, sorted filenames).
import torchvision.transforms as T
torch.manual_seed(42)
AUGMENTATIONS_PER_IMAGE = 4
AUGMENT = train_mod.AUGMENT

rows = []
for plant_dir in sorted(DATASET_DIR.iterdir()):
    if not plant_dir.is_dir():
        continue
    for f in sorted(plant_dir.glob("*.jpg")):
        rows.append((plant_dir.name, f))

X_mb_list = []
for label, f in rows:
    img = Image.open(f).convert("RGB")
    X_mb_list.append(train_mod.embed_image(model_mb, preprocess_mb, img))
    for _ in range(AUGMENTATIONS_PER_IMAGE):
        aug = AUGMENT(img)
        X_mb_list.append(train_mod.embed_image(model_mb, preprocess_mb, aug))

X_mb = np.stack(X_mb_list)
print(f"MobileNet embeddings shape: {X_mb.shape}, DINO embeddings shape: {X_dino.shape}")

assert X_mb.shape[0] == X_dino.shape[0], "row count mismatch - order assumption broken"

X_combined = np.concatenate([X_dino, X_mb], axis=1)
print(f"Combined feature dim: {X_combined.shape[1]}")

print("\n=== Classifiers on DINO + MobileNet concatenated ===")
for C in [0.1, 0.3, 1.0]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))
    mean, scores = cv_accuracy(clf, X_combined, y, groups)
    print(f"LogReg C={C}: mean={mean:.3f} scores={np.round(scores,3)}")

for hidden, alpha in [((256,), 0.01), ((128,), 0.01)]:
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=800, random_state=42),
    )
    mean, scores = cv_accuracy(clf, X_combined, y, groups)
    print(f"MLP hidden={hidden} alpha={alpha}: mean={mean:.3f} scores={np.round(scores,3)}")

np.save("combined_X.npy", X_combined)
