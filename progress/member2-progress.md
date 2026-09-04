# Member 2 Progress

## Role
Plant AI / Computer Vision

## Current Status
Real plant identification is implemented, wired into the existing backend
contract, and tested end-to-end (server started, hit `/api/identify` over
HTTP with real files, not just unit-level calls). It replaces Member 4's
mock `plant_ai.py` placeholder without changing `routes/identify.py` or
any response fields.

## Model Approach
Transfer learning via frozen feature extraction: MobileNetV3-Small
(ImageNet-pretrained, frozen) produces a 576-d embedding per image; a
Logistic Regression classifier trained on the curated 8 plants sits on top.
Chosen over full fine-tuning because the dataset is small (~18-20 images/
class) and would overfit a fully-trained CNN. See `ml/README.md` for full
detail and the reasoning behind every decision below.

## Supported Plants
All 8 plants already seeded in `backend/data/plants.json` by Member 4:
plant_01 (Indian Laburnum) through plant_08 (Curry Leaf Tree). No plant_id
changes.

## Completed
- `ml/fetch_dataset.py`: bootstraps `ml/dataset/<plant_id>/` with ~18-20
  images per plant from Wikimedia Commons (searched by scientific name).
  **Placeholder data, not real campus photos** — see Problems below.
- Manually reviewed a sample of downloaded images per class (not just
  counted them) and found/removed one clearly wrong image in `plant_06`
  (Mango): a Commons search hit that was a photo of a person eating a
  mango, not the plant. Everything else checked was correctly on-species.
- `ml/train.py`: extracts embeddings, trains the classifier, reports
  cross-validated accuracy, saves the model artifacts.
- `ml/inference.py`: loads the trained model and exposes
  `identify_plant(image_bytes) -> {"plant_id": str|None, "confidence": float}`
  — the exact contract the mock used.
- Added an out-of-distribution guard after finding it was needed (see
  Problems/Decisions) — rejects inputs that don't resemble any training
  image via cosine-similarity to reference embeddings, independent of the
  classifier's own confidence.
- Rewrote `backend/services/plant_ai.py` to call the real model, with an
  automatic fallback to the original mock if ML dependencies or model
  files aren't available (protects the backend from crashing if `torch`
  isn't installed in some environment — e.g. before a teammate has run
  `pip install -r requirements.txt` with the new deps).
- Added `torch`, `torchvision` (CPU wheels via `--extra-index-url`),
  `scikit-learn`, `joblib` to `backend/requirements.txt`, pinned to the
  exact versions verified working locally.

## Files Changed
- `ml/fetch_dataset.py` (new)
- `ml/train.py` (new)
- `ml/inference.py` (new)
- `ml/README.md` (new — full write-up of approach, accuracy, limitations)
- `ml/dataset/plant_01/` … `plant_08/` (new — ~150 images total, ~47MB)
- `ml/model/classifier.joblib`, `classes.json`, `train_embeddings.npy` (new
  — trained artifacts, committed so the backend works without a retrain
  step)
- `backend/services/plant_ai.py` (rewritten internals only — same
  `identify_plant(image_bytes)` signature and return contract; route file
  `routes/identify.py` untouched)
- `backend/requirements.txt` (added ML dependencies)

## API Status
`POST /api/identify` unchanged externally. Verified locally with the
backend actually running (`uvicorn main:app`), via real HTTP requests
(`curl`), not just function calls:
- Real images of all 8 plants → correctly identified (note: these were
  training images, so this only confirms wiring/plumbing works end-to-end,
  **not** real-world accuracy — see Accuracy/Testing below for the number
  that actually matters).
- Empty upload → `{"plant_id": null, ..., "error": "Uploaded image was empty"}`
- Non-image bytes → `{"plant_id": null, ..., "error": "Plant could not be identified"}`
- Random-noise image → correctly rejected as unidentified (only *after*
  adding the novelty guard — see Problems).
- Real photo of an unsupported species (rose, not one of the 8) →
  correctly rejected (confidence 0.43, below threshold).
- Unknown/malformed requests still handled entirely by
  `routes/identify.py`/`plant_service.py`, untouched by this work.

## Accuracy / Testing
**Honest number, from actual cross-validation, not a guess:** 4-fold
stratified CV accuracy = **~62%** on the current (placeholder) dataset.
At the 0.6 confidence threshold, accuracy among accepted predictions is
~76% at ~64% coverage — i.e. roughly a third of the time it correctly
says "not confident" instead of guessing.

Weakest classes: plant_03 (Peepal) vs plant_04 (Banyan) — both aerial-
rooted Ficus, confused with each other most often; plant_06 (Mango) — high
intra-class visual variance in the reference photos (whole tree / flowers
/ fruit close-ups don't look alike). Full confusion matrix and per-class
precision/recall are in the terminal output of `ml/train.py` if reproduced
locally; not duplicating the raw numbers here to avoid this file going
stale the next time the model is retrained.

I did not test on physical/live camera photos of real plants — that
requires either real campus photos or Member 3's camera capture flow to
be ready. Member 5: this dataset and these numbers need your independent
QA pass per your test matrix, especially the "unsupported plant" and
"low-confidence" cases in your prompt — I only spot-checked those myself.

## Problems
1. **Dataset is placeholder Wikimedia reference images, not real campus
   photos.** Mixes studio/product shots with field photos; this is very
   likely suppressing accuracy below what real, consistently-shot campus
   photos would achieve. Team decision made with the user: use public
   reference images now to unblock the pipeline, swap in real photos
   later if time allows before the demo.
2. **Found and fixed a real bug during testing, not just in review:** the
   classifier was initially confidently wrong on out-of-distribution
   input — a random-noise image scored 0.87 confidence as "Neem," well
   past the confidence threshold. This is exactly the "non-plant image"
   failure mode the master prompt calls out. Fixed with a cosine-
   similarity novelty guard (see `ml/README.md` "Out-of-distribution
   guard"); reverified after the fix that noise is now rejected and real
   plant images still work.
3. **Deployment size risk:** `torch`+`torchvision` are meaningfully larger
   than the rest of the backend's dependencies. Flagged to Member 1 in
   `ml/README.md` — worth a deploy dry run on Render/Railway sooner
   rather than the night before the demo, since the mock fallback covers
   "doesn't crash" but not "real identification still works."

## Decisions
- Frozen pretrained feature extraction + linear classifier, not
  fine-tuning: matches dataset size, avoids overfitting, faster to train/
  retrain if data changes.
- MobileNetV3-Small over a larger backbone: tried ResNet18 (512-d
  embeddings) side-by-side — no meaningful accuracy improvement (62% vs
  62%) — so kept the smaller/lighter model for deployment.
- Confidence threshold kept at 0.6 (matching the original mock's
  threshold) for consistency across the mock→real swap.
- Added the novelty/out-of-distribution guard as a second, independent
  gate rather than trying to fix it by only raising the confidence
  threshold — raising the threshold wouldn't have reliably caught the
  noise case (see `ml/README.md` for the calibration numbers).
- `plant_ai.py` falls back to the original mock automatically if ML deps/
  model files are missing, rather than raising and crashing the backend —
  reliability over completeness, per the master prompt's priority order.

## Next Tasks
- Coordinate with Member 5 on getting real campus photos of the 8 species
  to retrain against (`python ml/train.py` after repopulating
  `ml/dataset/`) — expected to meaningfully improve accuracy over the
  current placeholder dataset.
- Coordinate with Member 1 on a Render/Railway deploy dry run given the
  torch/torchvision dependency size.
- If time allows: targeted data collection specifically for plant_03/
  plant_04 (Peepal vs Banyan) and plant_06 (Mango), the three weakest
  classes.

## Handoff Notes
- **How to start the service:** it's part of the normal backend startup —
  `cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000`.
  No separate process; `plant_ai.py` loads the model once at import time.
- **What endpoint to call:** unchanged — `POST /api/identify`,
  multipart `image` field.
- **What response is returned:** unchanged contract —
  `{plant_id, name, scientific_name, confidence}` or the `error` variant.
- **What model/dependency is required:** `torch`, `torchvision`,
  `scikit-learn`, `joblib` (now in `backend/requirements.txt`). Falls back
  to the deterministic mock automatically if these aren't installed or
  `ml/model/*` is missing — check startup logs for a "falling back to
  mock" warning.
- **What remains imperfect:** ~62% cross-validated accuracy on a
  placeholder (non-campus) dataset; Peepal/Banyan and Mango are the
  weakest classes; real campus photos would likely help substantially.
