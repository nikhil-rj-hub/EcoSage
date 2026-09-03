"""
Pydantic schemas shared across the backend.

These mirror the canonical API contracts in docs/API.md exactly.
Do NOT rename, remove, or add required fields without updating the
contract doc and notifying the team (see project root instructions).
"""

from typing import List, Optional

from pydantic import BaseModel


class Plant(BaseModel):
    """Canonical plant object as stored in backend/data/plants.json."""

    plant_id: str
    common_name: str
    scientific_name: str
    family: str
    native_region: str
    ecological_importance: str
    conservation_status: str
    threats: List[str] = []
    conservation_actions: List[str] = []
    description: str
    image: str


class PlantSummary(BaseModel):
    """Lightweight plant entry used in the /api/plants list response."""

    plant_id: str
    common_name: str
    scientific_name: str


class PlantListResponse(BaseModel):
    plants: List[PlantSummary]


class IdentifyResponse(BaseModel):
    """Response for POST /api/identify."""

    plant_id: Optional[str] = None
    name: Optional[str] = None
    scientific_name: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""

    plant_id: str
    question: str


class ChatResponse(BaseModel):
    """Response for POST /api/chat."""

    answer: str


class HealthResponse(BaseModel):
    status: str
