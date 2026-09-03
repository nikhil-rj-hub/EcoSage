# Member 4 Progress

## Role
Backend / Plant Data APIs / AI Botanical Assistant

## Current Status
Full backend is bootstrapped and working end-to-end. All 5 required endpoints
(`/api/health`, `/api/plants`, `/api/plants/{plant_id}`, `/api/chat`,
`/api/identify`) are implemented, tested locally, and match the canonical
API contracts exactly. The repo was completely empty at the start of this
session, so this also stood up the base `backend/` structure.

Note: Member 2's plant AI (`services/plant_ai.py`) and its route
(`routes/identify.py`) are currently **mock placeholders**, clearly marked
as such in-file, so the rest of the system could be integrated and tested
now. Member 2 should replace the internals of `plant_ai.identify_plant`
with the real classifier — the function signature/return contract
(`{"plant_id": str|None, "confidence": float}`) and the route wiring
should stay the same so nothing else needs to change.

Similarly, `backend/data/plants.json` (Member 5's file) was seeded with 8
plants (real, commonly-documented Indian-subcontinent species suited to a
campus setting) so the data-dependent endpoints and AI context had
something real to work against. Member 5 should verify/replace facts and
add actual images to `public/plant-images/`.

## Backend Completed
- FastAPI app (`backend/main.py`) with CORS configured via `ALLOWED_ORIGINS`.
- Pydantic schemas for every contract: `Plant`, `PlantSummary`,
  `PlantListResponse`, `IdentifyResponse`, `ChatRequest`, `ChatResponse`,
  `HealthResponse` (`backend/models/schemas.py`).
- `plant_service.py`: loads/caches `data/plants.json`, exposes
  `list_plants()`, `get_plant(id)`, `plant_exists(id)`, and
  `build_botanical_context(id)` (renders trusted plant facts as text for
  LLM grounding).
- `botanical_ai.py`: builds a system prompt that instructs the LLM to rely
  only on the supplied plant context and to say "I don't know" rather than
  invent facts; calls Anthropic's API if `LLM_API_KEY` is set, otherwise
  falls back to a deterministic, context-derived answer.
- Routes: `routes/plants.py`, `routes/chat.py`, `routes/identify.py`
  (mock), all thin — they validate input and delegate to services, no
  business logic or hardcoded plant data in route functions.
- Seed data: `backend/data/plants.json` with 8 plants (plant_01–plant_08),
  matching the canonical schema field-for-field.
- Mock `services/plant_ai.py`: deterministic (hash-based) fake
  classifier so demos are repeatable; respects a confidence threshold and
  returns "not identified" below it.

## APIs
- `GET /api/health` -> `{"status": "ok"}`
- `GET /api/plants` -> `{"plants": [{plant_id, common_name, scientific_name}, ...]}`
- `GET /api/plants/{plant_id}` -> full `Plant` object, 404 if unknown
- `POST /api/chat` -> `{"plant_id", "question"}` in, `{"answer"}` out;
  404 if plant_id unknown, 400 if question is empty/whitespace
- `POST /api/identify` -> multipart `image` field in, `IdentifyResponse`
  out (plant_id/name/scientific_name/confidence, or `error` set with
  confidence 0.0 on failure)

No fields were renamed or removed from the contracts in the master prompt.

## AI Assistant
- Model: Anthropic (`LLM_MODEL`, default `claude-haiku-4-5-20251001`) via
  the `anthropic` Python SDK.
- Grounding: `plant_id -> plant_service.build_botanical_context() ->
  system+user prompt -> LLM -> answer`. The system prompt explicitly
  forbids inventing facts not in the context.
- Fallback (no API key, or LLM call throws/times out): returns a
  deterministic answer built directly from the trusted plant context, so
  chat never hard-fails. `LLM_TIMEOUT_SECONDS` (default 15s) bounds the
  call.

## Files Changed
New files (repo was empty before this session):
- `backend/main.py`
- `backend/requirements.txt`
- `backend/.env.example`
- `backend/models/__init__.py`, `backend/models/schemas.py`
- `backend/routes/__init__.py`, `backend/routes/plants.py`,
  `backend/routes/chat.py`, `backend/routes/identify.py` (placeholder)
- `backend/services/__init__.py`, `backend/services/plant_service.py`,
  `backend/services/botanical_ai.py`, `backend/services/plant_ai.py` (placeholder)
- `backend/data/plants.json` (seed data, placeholder facts for Member 5 to verify)
- `.gitignore` (root)
- `progress/member4-progress.md` (this file)

## APIs / Interfaces Changed
N/A — first implementation, contracts as defined in the master prompt.

## Dependencies Added
`backend/requirements.txt`:
- fastapi==0.115.0
- uvicorn[standard]==0.30.6
- pydantic==2.9.2
- python-multipart==0.0.9 (required for multipart/form-data image upload)
- anthropic==0.34.2 (Botanical AI LLM calls)

## Integration Notes
- Backend startup: from `backend/`, `pip install -r requirements.txt`
  then `uvicorn main:app --reload --port 8000`.
- Env vars (see `.env.example`): `LLM_API_KEY` (optional — falls back
  gracefully if unset), `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`,
  `ALLOWED_ORIGINS` (CORS; must include the frontend dev URL, default
  `http://localhost:5173`).
- Frontend can build against all 5 endpoints right now — the identify
  mock returns a real, valid plant from `plants.json` deterministically
  keyed off the uploaded image bytes, so repeated uploads of the same
  image give consistent demo results.
- Member 2: replace `services/plant_ai.py::identify_plant` only; don't
  touch `routes/identify.py` unless the contract itself needs to change
  (in which case, coordinate first).
- Member 5: replace `backend/data/plants.json` content/facts and add real
  files under `public/plant-images/` matching the `image` paths already
  referenced (`/plant-images/plant_01.jpg` ... `plant_08.jpg`).

## Problems / Blockers
- None currently blocking. Real plant identification accuracy is
  obviously untested since it's still a mock — that's Member 2's task.
- No LLM key was available in this environment, so the chat fallback
  path is what's been exercised; the real Anthropic call path is
  implemented per their SDK but should be smoke-tested once a key is
  provided (either by me or whoever holds the team's key).

## Decisions Made
- Used Anthropic's Claude API (via `anthropic` SDK) for the Botanical AI,
  since that's directly supported/documented in this environment and
  satisfies the "LLM API" requirement without adding a new provider
  dependency risk.
- Chat failures (LLM down/timeout) return HTTP 200 with a graceful
  fallback `answer` rather than a 5xx, so the UI never has to
  special-case "AI is down" beyond just rendering the text — matches the
  "never leave the user staring at a blank screen" rule. Unknown
  plant_id / empty question are still proper 404/400 errors since those
  are client mistakes, not AI failures.
- Seeded `plants.json` with 8 real, verifiable-facts plants (not 7) to
  give a small buffer and one genuine conservation story (Ashoka tree,
  IUCN Vulnerable) for the "conservation awareness" feature to showcase.
- Added `__init__.py` files to `models/`, `routes/`, `services/` for
  explicit, reliable imports regardless of how the app is launched.

## Testing Performed
Ran the server locally (`uvicorn main:app`) and hit every endpoint with
curl:
- [x] `GET /api/health` -> 200 `{"status":"ok"}`
- [x] `GET /api/plants` -> 200, all 8 plants listed with correct summary fields
- [x] `GET /api/plants/plant_01` -> 200, full schema, all fields present
- [x] `GET /api/plants/plant_99` -> 404 `{"detail":"Plant 'plant_99' not found"}`
- [x] `POST /api/chat` valid plant_id+question -> 200, fallback answer built
      correctly from that plant's real context (verified Ashoka tree data
      appeared correctly)
- [x] `POST /api/chat` unknown plant_id -> 404
- [x] `POST /api/chat` empty/whitespace question -> 400
- [x] `POST /api/chat` missing `question` field -> 422 (Pydantic validation)
- [x] `POST /api/identify` with a valid file -> 200, returned a real
      plant_id/name/scientific_name/confidence from plants.json
- [x] `POST /api/identify` with no file -> 422
- [x] `POST /api/identify` with an empty file -> 200,
      `{"plant_id":null,...,"error":"Uploaded image was empty"}`
- [x] Confirmed server starts cleanly and stops cleanly (no orphan process
      left running after testing)

Not yet tested: real Anthropic LLM call path (no API key in this
environment), frontend integration (Member 3's side), real ML
identification (Member 2's side).

## Next Tasks
- Smoke-test the real LLM call path once an `LLM_API_KEY` is available.
- Coordinate with Member 2 on swapping in the real `plant_ai.py`.
- Coordinate with Member 5 on final `plants.json` facts + images.
- Add a lightweight automated test file (pytest) if time permits — all
  testing so far was manual via curl.

## Handoff Notes (for Member 1)
- **Startup command:** `cd backend && pip install -r requirements.txt &&
  uvicorn main:app --reload --port 8000`
- **Env vars needed:** `LLM_API_KEY` (optional, has a working fallback),
  `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`, `ALLOWED_ORIGINS` — see
  `backend/.env.example`.
- **Endpoints live:** `/api/health`, `/api/plants`, `/api/plants/{id}`,
  `/api/chat`, `/api/identify` — all match the master contract exactly,
  verified with curl (see Testing Performed above).
- **Current limitations:** `/api/identify` uses a deterministic mock
  classifier (not real ML) until Member 2 finishes; `plants.json` data
  should be fact-checked by Member 5 before the final demo.
- **No secrets committed** — `.env.example` only, real `.env` is
  gitignored.
