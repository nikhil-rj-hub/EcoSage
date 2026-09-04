# Plant Identification (Member 2)

Real implementation behind `POST /api/identify`. Replaces the mock that
previously lived in `backend/services/plant_ai.py`.

## Approach

Transfer learning via frozen feature extraction (not full fine-tuning):

```
image -> MobileNetV3-Small (ImageNet-pretrained, frozen) -> 576-d embedding
      -> Logistic Regression classifier (trained on curated plants)
      -> plant_id + confidence
```

Fine-tuning a full CNN was ruled out: the curated dataset only has
~18-20 images per class, which would badly overfit a network with millions
of trainable parameters. Frozen pretrained features + a simple linear
classifier on top is the standard, much more data-efficient approach for
this dataset size, and matches the master prompt's guidance to "start with
the simplest viable approach."

## Dataset

`ml/dataset/<plant_id>/*.jpg` — one folder per plant, currently populated
via `fetch_dataset.py`, which pulls freely-licensed reference photos from
Wikimedia Commons keyed on each plant's scientific name.

**This is a placeholder dataset for prototyping, not real campus photos.**
It was manually spot-checked (not just count-checked) and one clearly wrong
image was found and removed (a photo of a person eating a mango, returned
by a Commons search for "Mangifera indica"). The remaining images are
correctly on-species, but they mix studio/product shots with outdoor field
photos, which adds noise a more consistent, campus-photographed dataset
would not have.

**Action needed from Member 5 / the team:** replace this with real photos
of the actual campus specimens before the final demo if at all possible —
see "Known accuracy" below for why this matters.

## Known accuracy (measured, not assumed)

4-fold stratified cross-validation on the current dataset: **~62% accuracy**
(`python train.py` prints this each run). Confidence is reasonably
calibrated — correct predictions average ~0.79 confidence, incorrect ones
average ~0.60 — which is what the `CONFIDENCE_THRESHOLD = 0.6` gate in
`inference.py` relies on. At that threshold, cross-validated accuracy among
*accepted* (non-rejected) predictions is ~76%, at ~64% coverage.

The two weakest classes are `plant_03` (Peepal) and `plant_04` (Banyan) —
both are aerial-rooted Ficus trees that look genuinely similar in generic
reference photos, and they're confused with each other more than with
anything else. `plant_06` (Mango) is also weak, likely because its
reference images span very different views (whole tree, flower clusters,
fruit close-ups) that don't share much visual similarity. Real campus
photos of the *same* specimens (consistent lighting/framing) would very
likely score meaningfully higher than this generic internet dataset.

Do not quote a higher number than this in the presentation without
rerunning `train.py` against updated data and reporting what it prints.

## Out-of-distribution guard

A softmax classifier can be confidently wrong on inputs unlike anything it
was trained on. This was verified directly: before the guard below was
added, a random-noise image was classified as "Neem" at 0.87 confidence —
comfortably past the 0.6 threshold. The master prompt explicitly requires
handling a "non-plant image" test case; a confidence threshold on its own
does not.

Fix: in addition to the confidence threshold, `inference.py` rejects an
image if its embedding isn't at least `0.45` cosine-similar to *any*
training image (`NOVELTY_SIMILARITY_THRESHOLD` in `inference.py`). This
threshold was calibrated from the training set's own in-distribution
nearest-neighbor similarities (~0.54-0.9+) vs. random noise (~0.19) — 0.45
sits with margin below the in-distribution range. Verified after adding it:
random noise -> rejected (`plant_id: null`); a real photo of an unsupported
species (a rose) -> also correctly rejected by the confidence threshold.

This is a coarse guard, not a rigorous OOD detector — it will not catch
every possible non-plant input, but it closes the specific, verified
failure mode above.

## Files

- `fetch_dataset.py` — one-off script to bootstrap `dataset/` from Wikimedia
  Commons. Re-run per-plant if a class needs more/better images.
- `train.py` — extracts embeddings, trains the classifier, prints
  cross-validated accuracy, saves `model/classifier.joblib`,
  `model/classes.json`, and `model/train_embeddings.npy`.
- `inference.py` — loads the trained artifacts and exposes
  `identify_plant(image_bytes) -> {"plant_id": str|None, "confidence": float}`,
  the same contract the mock it replaces used.
- `model/` — trained artifacts (regenerate by running `train.py`; not
  meaningfully useful without `dataset/`, but small enough to commit so the
  backend works without a retrain step).

## Retraining

```
cd ml
pip install -r ../backend/requirements.txt   # torch/torchvision/scikit-learn/joblib
python fetch_dataset.py   # only if you need to (re)populate dataset/
python train.py
```

Restart the backend afterward — `plant_ai.py` loads the model once at
import time, not per-request.

## Integration

`backend/services/plant_ai.py` imports this package (via a `sys.path`
insert to the repo-root `ml/` directory, since `ml/` is a sibling of
`backend/`, not a subpackage of it) and calls `identify_plant` directly.
**If `torch`/`torchvision`/`scikit-learn` aren't installed, or the model
files are missing, `plant_ai.py` automatically falls back to the original
deterministic mock** rather than crashing the backend — see the comment at
the top of that file. This means the backend stays runnable even in an
environment where the (fairly large) ML dependencies haven't been
installed, at the cost of silently degrading to the mock; check the
backend's startup logs for a "falling back to mock" warning to know which
mode is active.

## Deployment note for Member 1

`torch` + `torchvision` (CPU wheels) add real weight to
`backend/requirements.txt` — noticeably bigger than the app's other
dependencies. Flagging this now in case it affects the free-tier
Render/Railway build: if the build fails on size/time, the fallback mock
above keeps the rest of the app working while we sort it out, but real
identification would be down. Worth a build/deploy dry run earlier rather
than the night before the demo.
