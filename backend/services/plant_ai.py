"""
Plant identification AI service.

*** PLACEHOLDER / MOCK IMPLEMENTATION ***
This file is owned by Member 2 (Plant AI / Computer Vision). It is
stubbed here ONLY so the backend and frontend can be developed and
integrated end-to-end before the real model is ready, per the project's
mocking rule. Member 2 should replace `identify_plant` with a real
implementation while preserving its return contract below.

Return contract (must be preserved by the real implementation):

    identify_plant(image_bytes: bytes) -> dict with keys:
        plant_id: str | None
        confidence: float   (0.0 - 1.0)

    Returns plant_id=None, confidence=0.0 when identification fails
    or confidence would be below CONFIDENCE_THRESHOLD.
"""

import hashlib
from typing import Optional, TypedDict

from services import plant_service

CONFIDENCE_THRESHOLD = 0.6


class IdentificationResult(TypedDict):
    plant_id: Optional[str]
    confidence: float


def identify_plant(image_bytes: bytes) -> IdentificationResult:
    """
    MOCK: deterministically "identifies" a plant from image bytes so the
    same uploaded image always returns the same result during demos.

    Not a real classifier - do not use for anything beyond integration
    testing and UI development.
    """
    if not image_bytes:
        return {"plant_id": None, "confidence": 0.0}

    known_plant_ids = [p.plant_id for p in plant_service.list_plants()]
    if not known_plant_ids:
        return {"plant_id": None, "confidence": 0.0}

    digest = hashlib.sha256(image_bytes).hexdigest()
    index = int(digest, 16) % len(known_plant_ids)
    # Deterministic mock confidence in the 0.75-0.97 range.
    mock_confidence = 0.75 + (int(digest[:4], 16) % 23) / 100

    if mock_confidence < CONFIDENCE_THRESHOLD:
        return {"plant_id": None, "confidence": mock_confidence}

    return {"plant_id": known_plant_ids[index], "confidence": round(mock_confidence, 2)}
