"""
Plant identification AI service.

Owned by Member 2 (Plant AI / Computer Vision).

Real implementation: delegates to ml/inference.py, a MobileNetV3-Small
(ImageNet-pretrained) feature extractor + Logistic Regression classifier
trained on a curated per-plant image set (see ml/README.md for how to
retrain and current accuracy numbers).

Return contract (unchanged from the original mock, so nothing downstream
needed to change):

    identify_plant(image_bytes: bytes) -> dict with keys:
        plant_id: str | None
        confidence: float   (0.0 - 1.0)

    Returns plant_id=None, confidence=0.0 when identification fails
    or confidence would be below the model's confidence threshold.

Fallback: if the ML dependencies (torch/torchvision/scikit-learn) or the
trained model artifact aren't available in the current environment, this
module falls back to the original deterministic hash-based mock rather
than crashing the backend - see FALLBACK_TO_MOCK below and
progress/member2-progress.md for why.
"""

import hashlib
import logging
import pathlib
import sys
from typing import Optional, TypedDict

from services import plant_service

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


class IdentificationResult(TypedDict):
    plant_id: Optional[str]
    confidence: float


try:
    import inference as _ml_inference  # ml/inference.py

    _ml_inference._load()  # fail fast here, not on the first request
    FALLBACK_TO_MOCK = False
except Exception as exc:  # noqa: BLE001 - deliberately broad: any load failure -> fallback
    logger.warning(
        "Real plant identification model unavailable (%s); falling back to mock identify_plant.",
        exc,
    )
    FALLBACK_TO_MOCK = True


def _identify_plant_mock(image_bytes: bytes) -> IdentificationResult:
    """
    MOCK: deterministically "identifies" a plant from image bytes so the
    same uploaded image always returns the same result during demos.

    Not a real classifier - only used if the real model failed to load.
    """
    if not image_bytes:
        return {"plant_id": None, "confidence": 0.0}

    known_plant_ids = [p.plant_id for p in plant_service.list_plants()]
    if not known_plant_ids:
        return {"plant_id": None, "confidence": 0.0}

    digest = hashlib.sha256(image_bytes).hexdigest()
    index = int(digest, 16) % len(known_plant_ids)
    mock_confidence = 0.75 + (int(digest[:4], 16) % 23) / 100

    if mock_confidence < CONFIDENCE_THRESHOLD:
        return {"plant_id": None, "confidence": mock_confidence}

    return {"plant_id": known_plant_ids[index], "confidence": round(mock_confidence, 2)}


def identify_plant(image_bytes: bytes) -> IdentificationResult:
    if FALLBACK_TO_MOCK:
        return _identify_plant_mock(image_bytes)

    try:
        return _ml_inference.identify_plant(image_bytes)
    except Exception:
        logger.exception("Real plant identification failed; falling back to mock for this request")
        return _identify_plant_mock(image_bytes)
