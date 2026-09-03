"""
Route for POST /api/identify.

*** PLACEHOLDER ***
This file is owned by Member 2 (Plant AI / Computer Vision). It is
stubbed here only to unblock frontend/backend integration ahead of the
real model (see services/plant_ai.py for details). The route wiring
(multipart upload -> plant_ai.identify_plant -> plant_service lookup ->
IdentifyResponse) should remain stable even after Member 2 swaps in the
real classifier - only the internals of plant_ai.identify_plant need to
change.
"""

import logging

from fastapi import APIRouter, File, UploadFile

from models.schemas import IdentifyResponse
from services import plant_ai, plant_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["identify"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/identify", response_model=IdentifyResponse)
async def identify(image: UploadFile = File(...)) -> IdentifyResponse:
    try:
        image_bytes = await image.read()
    except Exception:
        logger.exception("Failed to read uploaded image")
        return IdentifyResponse(error="Could not read uploaded image")

    if not image_bytes:
        return IdentifyResponse(error="Uploaded image was empty")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return IdentifyResponse(error="Image too large")

    try:
        result = plant_ai.identify_plant(image_bytes)
    except Exception:
        logger.exception("Plant identification service failed")
        return IdentifyResponse(error="Plant could not be identified")

    plant_id = result.get("plant_id")
    confidence = result.get("confidence", 0.0)

    if not plant_id:
        return IdentifyResponse(confidence=confidence, error="Plant could not be identified")

    plant = plant_service.get_plant(plant_id)
    if plant is None:
        # Identification returned an id that isn't in the knowledge base.
        logger.error("plant_ai returned unknown plant_id=%s", plant_id)
        return IdentifyResponse(error="Plant could not be identified")

    return IdentifyResponse(
        plant_id=plant.plant_id,
        name=plant.common_name,
        scientific_name=plant.scientific_name,
        confidence=confidence,
    )
