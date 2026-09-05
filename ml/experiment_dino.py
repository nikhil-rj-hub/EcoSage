"""
One-off experiment: does DINO ViT-Small give better embeddings than
MobileNetV3-Small for this dataset? Compared honestly, same way ResNet18
was compared earlier (side-by-side, same CV methodology, no cherry-picking).

Also compares classifier heads (Logistic Regression, k-NN, small MLP) on
top of whichever embedding wins, since that's a separate, cheap lever.
"""

import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
import torchvision.transforms as T

DATASET_DIR = pathlib.Path(__file__).parent / "dataset"

# Same augmentation as train.py, for a fair comparison.
AUGMENTATIONS_PER_IMAGE = 4
AUGMENT = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
    T.RandomRotation(degrees=12),
])


def build_dino():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def embed(model, preprocess, image: Image.Image) -> np.ndarray:
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze(0).numpy()


def build_dataset(model, preprocess):
    X, y, groups = [], [], []
    gid = 0
    for plant_dir in sorted(DATASET_DIR.iterdir()):
        if not plant_dir.is_dir():
            continue
        image_files = sorted(plant_dir.glob("*.jpg"))
        print(f"[{plant_dir.name}] embedding {len(image_files)} images...")
        for f in image_files:
            img = Image.open(f).convert("RGB")
            X.append(embed(model, preprocess, img))
            y.append(plant_dir.name)
            groups.append(gid)
            for _ in range(AUGMENTATIONS_PER_IMAGE):
                aug = AUGMENT(img)
                X.append(embed(model, preprocess, aug))
                y.append(plant_dir.name)
                groups.append(gid)
            gid += 1
    return np.stack(X), np.array(y), np.array(groups)


def cv_accuracy(clf, X, y, groups):
    photos_per_class = {}
    for label, gid in zip(y, groups):
        photos_per_class.setdefault(label, set()).add(gid)
    min_photos = min(len(v) for v in photos_per_class.values())
    n_splits = max(2, min(4, min_photos))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=sgkf, groups=groups)
    return scores.mean(), scores


def main():
    print("Loading DINO ViT-Small...")
    model, preprocess = build_dino()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nEmbedding dataset with DINO...")
    X, y, groups = build_dataset(model, preprocess)
    print(f"\nTotal samples: {len(y)}, feature dim: {X.shape[1]}")

    np.save("dino_X.npy", X)
    np.save("dino_y.npy", y)
    np.save("dino_groups.npy", groups)

    print("\n=== Classifier comparison on DINO embeddings ===")

    clf_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3))
    mean, scores = cv_accuracy(clf_lr, X, y, groups)
    print(f"Logistic Regression (C=0.3): mean={mean:.3f} scores={np.round(scores, 3)}")

    for C in [0.05, 0.1, 1.0]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))
        mean, scores = cv_accuracy(clf, X, y, groups)
        print(f"Logistic Regression (C={C}): mean={mean:.3f}")

    for k in [3, 5, 9, 15]:
        clf_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        mean, scores = cv_accuracy(clf_knn, X, y, groups)
        print(f"k-NN (k={k}): mean={mean:.3f} scores={np.round(scores, 3)}")

    clf_mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128,), alpha=0.01, max_iter=500, random_state=42),
    )
    mean, scores = cv_accuracy(clf_mlp, X, y, groups)
    print(f"MLP (128 hidden, alpha=0.01): mean={mean:.3f} scores={np.round(scores, 3)}")


if __name__ == "__main__":
    main()
