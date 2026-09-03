"""Routes for GET /api/plants and GET /api/plants/{plant_id}."""

from fastapi import APIRouter, HTTPException

from models.schemas import Plant, PlantListResponse
from services import plant_service

router = APIRouter(prefix="/api", tags=["plants"])


@router.get("/plants", response_model=PlantListResponse)
def get_all_plants() -> PlantListResponse:
    return PlantListResponse(plants=plant_service.list_plants())


@router.get("/plants/{plant_id}", response_model=Plant)
def get_plant(plant_id: str) -> Plant:
    plant = plant_service.get_plant(plant_id)
    if plant is None:
        raise HTTPException(status_code=404, detail=f"Plant '{plant_id}' not found")
    return plant
