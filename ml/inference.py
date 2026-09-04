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
import json
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
CLASS_REFERENCE_EMBEDDINGS_PATH = MODEL_DIR / "class_reference_embeddings.npz"
CLASS_SIMILARITY_THRESHOLDS_PATH = MODEL_DIR / "class_similarity_thresholds.json"

CONFIDENCE_THRESHOLD = 0.6

# Out-of-distribution guard, v2 - see the matching comment in train.py for
# the full story of why this is per-class rather than one global threshold.
# In short: a single global "similar to anything we've trained on" check
# broke down once the dataset grew large and diverse (8/10 unsupported
# species were confidently misidentified). This version instead requires
# the query to be similar to photos of the SPECIFIC class the classifier
# predicted, using a threshold calibrated per-class from that class's own
# data (see class_similarity_thresholds.json, written by train.py).


class IdentificationResult(TypedDict):
    plant_id: Optional[str]
    confidence: float


_classifier = None
_backbone = None
_preprocess = None
_class_reference_embeddings = None
_class_similarity_thresholds = None


def _load() -> None:
    """Lazily load the model on first use (keeps backend import time fast)."""
    global _classifier, _backbone, _preprocess
    global _class_reference_embeddings, _class_similarity_thresholds
    if _classifier is not None:
        return

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"No trained classifier at {CLASSIFIER_PATH}. Run `python ml/train.py` first."
        )
    if not CLASS_REFERENCE_EMBEDDINGS_PATH.exists() or not CLASS_SIMILARITY_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"No novelty-guard reference data in {MODEL_DIR}. Run `python ml/train.py` first."
        )

    _classifier = joblib.load(CLASSIFIER_PATH)
    _class_reference_embeddings = dict(np.load(CLASS_REFERENCE_EMBEDDINGS_PATH))
    _class_similarity_thresholds = json.loads(
        CLASS_SIMILARITY_THRESHOLDS_PATH.read_text(encoding="utf-8")
    )

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

    Returns plant_id=None, confidence=0.0 if the image can't be decoded, if
    the model's top prediction confidence is below CONFIDENCE_THRESHOLD, or
    if the image doesn't resemble the predicted class closely enough to
    trust the prediction (out-of-distribution guard).
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

    probabilities = _classifier.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    predicted_id = str(_classifier.classes_[best_index])

    if confidence < CONFIDENCE_THRESHOLD:
        return {"plant_id": None, "confidence": round(confidence, 2)}

    normalized = (features / np.linalg.norm(features)).reshape(-1)
    class_embeddings = _class_reference_embeddings[predicted_id]
    similarity_to_predicted_class = float((class_embeddings @ normalized).max())
    threshold = _class_similarity_thresholds[predicted_id]
    if similarity_to_predicted_class < threshold:
        # Confident, but doesn't actually look like the predicted class's
        # own training photos - likely a species outside the curated set.
        return {"plant_id": None, "confidence": 0.0}

    return {"plant_id": predicted_id, "confidence": round(confidence, 2)}
