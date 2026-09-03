"""
Plant data service.

Owns loading and querying the plant knowledge base (backend/data/plants.json).
No other module should read plants.json directly - route handlers and the
botanical AI service must go through this module.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from models.schemas import Plant, PlantSummary

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "plants.json"

_lock = threading.Lock()
_plants_by_id: Optional[Dict[str, Plant]] = None


def _load() -> Dict[str, Plant]:
    """Load plants.json into memory once, cached for the process lifetime."""
    global _plants_by_id
    with _lock:
        if _plants_by_id is None:
            if not DATA_PATH.exists():
                _plants_by_id = {}
                return _plants_by_id
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            plants = [Plant(**p) for p in raw.get("plants", [])]
            _plants_by_id = {p.plant_id: p for p in plants}
        return _plants_by_id


def reload() -> None:
    """Force a reload of plants.json on next access. Useful for tests."""
    global _plants_by_id
    with _lock:
        _plants_by_id = None


def list_plants() -> List[PlantSummary]:
    plants = _load()
    return [
        PlantSummary(
            plant_id=p.plant_id,
            common_name=p.common_name,
            scientific_name=p.scientific_name,
        )
        for p in plants.values()
    ]


def get_plant(plant_id: str) -> Optional[Plant]:
    plants = _load()
    return plants.get(plant_id)


def plant_exists(plant_id: str) -> bool:
    return get_plant(plant_id) is not None


def build_botanical_context(plant_id: str) -> Optional[str]:
    """
    Render a plant's trusted data as a compact text block suitable for
    injection into an LLM prompt as grounding context.
    """
    plant = get_plant(plant_id)
    if plant is None:
        return None

    threats = "; ".join(plant.threats) if plant.threats else "Not documented"
    actions = (
        "; ".join(plant.conservation_actions)
        if plant.conservation_actions
        else "Not documented"
    )

    return (
        f"Common name: {plant.common_name}\n"
        f"Scientific name: {plant.scientific_name}\n"
        f"Family: {plant.family}\n"
        f"Native region: {plant.native_region}\n"
        f"Conservation status: {plant.conservation_status}\n"
        f"Ecological importance: {plant.ecological_importance}\n"
        f"Threats: {threats}\n"
        f"Conservation actions: {actions}\n"
        f"Description: {plant.description}"
    )
