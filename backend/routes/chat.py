"""Route for POST /api/chat - the AI Botanical Guide."""

import logging

from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse
from services import botanical_ai, plant_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not plant_service.plant_exists(request.plant_id):
        raise HTTPException(
            status_code=404, detail=f"Plant '{request.plant_id}' not found"
        )

    try:
        answer = botanical_ai.answer_question(request.plant_id, question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        logger.exception("Unexpected botanical AI failure for plant_id=%s", request.plant_id)
        answer = (
            "Sorry, I couldn't process that question right now. Please try again."
        )

    return ChatResponse(answer=answer)
