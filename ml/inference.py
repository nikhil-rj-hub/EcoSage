"""
Plant identification inference (Member 2 / Plant AI).

Loads the trained MobileNetV3-Small feature extractor + Logistic Regression
classifier (see train.py) and exposes a single function:

    identify_plant(image_bytes: bytes) -> dict with keys:
        plant_id: str | None
        confidence: float   (0.0 - 1.0)

This is the same return contract as the mock it replaces
(backend/services/plant_ai.py), so nothing downstream needs to change.
"""

import io
import pathlib
from typing import Optional, TypedDict

import joblib
import numpy as np
import torch
import torchvision
from PIL import Image, UnidentifiedImageError

HERE = pathlib.Path(__file__).parent
MODEL_DIR = HERE / "model"
CLASSIFIER_PATH = MODEL_DIR / "classifier.joblib"
TRAIN_EMBEDDINGS_PATH = MODEL_DIR / "train_embeddings.npy"

CONFIDENCE_THRESHOLD = 0.6

# Out-of-distribution guard: reject inputs whose embedding isn't at least
# this cosine-similar to *some* training image, even if the classifier is
# confident. Without this, a softmax classifier can be very confident on
# inputs unlike anything it was trained on (verified: random noise scored
# 0.87 confidence as "Neem" before this check was added). Calibrated from
# the training set's own leave-one-out nearest-neighbor similarities, which
# ranged ~0.54-0.9+ in-distribution vs ~0.19 for random noise - 0.45 sits
# safely below the in-distribution range with margin.
NOVELTY_SIMILARITY_THRESHOLD = 0.45


class IdentificationResult(TypedDict):
    plant_id: Optional[str]
    confidence: float


_classifier = None
_backbone = None
_preprocess = None
_train_embeddings = None


def _load() -> None:
    """Lazily load the model on first use (keeps backend import time fast)."""
    global _classifier, _backbone, _preprocess, _train_embeddings
    if _classifier is not None:
        return

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"No trained classifier at {CLASSIFIER_PATH}. Run `python ml/train.py` first."
        )
    if not TRAIN_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"No reference embeddings at {TRAIN_EMBEDDINGS_PATH}. Run `python ml/train.py` first."
        )

    _classifier = joblib.load(CLASSIFIER_PATH)
    _train_embeddings = np.load(TRAIN_EMBEDDINGS_PATH)

    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    backbone = torchvision.models.mobilenet_v3_small(weights=weights)
    backbone.classifier = torch.nn.Identity()
    backbone.eval()

    _backbone = backbone
    _preprocess = weights.transforms()


def _embed(image: Image.Image) -> np.ndarray:
    tensor = _preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = _backbone(tensor)
    return features.squeeze(0).numpy().reshape(1, -1)


def identify_plant(image_bytes: bytes) -> IdentificationResult:
    """
    Classify a plant image among the curated set the model was trained on.

    Returns plant_id=None, confidence=0.0 if the image can't be decoded, or
    if the model's top prediction confidence is below CONFIDENCE_THRESHOLD.
    """
    if not image_bytes:
        return {"plant_id": None, "confidence": 0.0}

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        return {"plant_id": None, "confidence": 0.0}

    _load()

    features = _embed(image)

    normalized = features / np.linalg.norm(features)
    novelty_similarity = float((_train_embeddings @ normalized.T).max())
    if novelty_similarity < NOVELTY_SIMILARITY_THRESHOLD:
        # Doesn't resemble anything the model was trained on (e.g. a
        # non-plant photo) - don't let the classifier guess anyway.
        return {"plant_id": None, "confidence": 0.0}

    probabilities = _classifier.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    predicted_id = str(_classifier.classes_[best_index])

    if confidence < CONFIDENCE_THRESHOLD:
        return {"plant_id": None, "confidence": round(confidence, 2)}

    return {"plant_id": predicted_id, "confidence": round(confidence, 2)}
