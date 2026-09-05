# Member 2 Progress

## Role
Plant AI / Computer Vision

## Current Status
Real plant identification is implemented, wired into the existing backend
contract, and tested end-to-end (server started, hit `/api/identify` over
HTTP with real files, not just unit-level calls). It replaces Member 4's
mock `plant_ai.py` placeholder without changing `routes/identify.py` or
any response fields.

**Final honest numbers: ~81.7-81.8% cross-validated accuracy (up from a
corrected 71.0% after fixing a duplicate-data bug — see below), and an
out-of-distribution guard that correctly rejects 7/10 confirmed
non-project-species test photos (up from 6/10).** This session had two
distinct corrections worth understanding in order:
1. Mid-session, an inflated 87.2% accuracy / 9/10 OOD number was found to
   be the result of a duplicate-data bug and corrected down to 71.0% / 6/10
   — see "The duplicate-content bug" below.
2. After that correction, the user asked what else could close the
   remaining gap to 90%. Switching the model backbone from a supervised
   CNN (MobileNetV3) to a self-supervised one (DINO ViT-Small) on the
   *same, already-corrected* dataset produced a second, real jump:
   71.0% -> 81.7-81.8% accuracy, 6/10 -> 7/10 OOD rejection. This is not
   a reversal of the correction — it's a genuine improvement measured
   honestly on top of the corrected baseline.

## The duplicate-content bug (read this first)
Partway through this session, after several rounds of "expand the
dataset, retrain, accuracy goes up," a routine integrity check (hashing
every image file) found that **553 of the dataset's 1340 files were
exact-duplicate content** under different filenames. Root cause:
`fetch_dataset.py` only tracked "already downloaded this run" in memory,
reset per invocation — so separate fetch runs (and I ran many, expanding
different classes at different times) kept re-downloading and re-saving
the same Commons photos under new filenames. This wasn't just wasted disk
space: `train.py`'s `StratifiedGroupKFold` was designed to keep all
*augmented copies* of one photo in the same cross-validation fold, but it
had no way to know that two different *files* were byte-identical, so
duplicate raw files could (and did) land in different folds — letting the
model be validated on near-copies of its own training data. This
inflated every accuracy number reported earlier in this session.

**Fixed:** wrote `dedupe_dataset.py` to remove the 553 duplicates, fixed
`fetch_dataset.py` to hash against every file already on disk before
saving a new one (not just this run's own history), then did one more
careful expansion round with the fix in place and retrained. Also
re-verified the out-of-distribution guard on the corrected model — it got
noticeably weaker (9/10 → 6/10 on the same 10-photo test set), which
makes sense: its per-class thresholds are calibrated from each class's
own photos, and the honest, smaller dataset gives a less precise estimate
of "how similar do this species' photos look to each other" than the
duplicate-padded one did.

## Model Approach
Transfer learning via frozen feature extraction: DINO ViT-Small
(self-supervised, frozen) produces a 384-d embedding per image; an MLP
classifier trained on the curated 8 plants sits on top, with 4x light
augmentation per photo. Chosen over full fine-tuning because the dataset
is small relative to the backbone's parameter count and would overfit.

**Backbone history (two rounds of honest comparison, not one guess):**
1. MobileNetV3-Small vs. ResNet18 (both supervised ImageNet CNNs) —
   identical accuracy (62% vs 62%). Conclusion at the time: architecture
   wasn't the bottleneck among supervised CNNs.
2. Later, MobileNetV3-Small vs. DINO ViT-Small (self-supervised, a
   fundamentally different training objective) — real jump, 71.0% ->
   81.7-81.8%, cross-validated the same way both times. Also tried
   concatenating DINO + MobileNetV3 features, which made things *worse*
   (79-80.5%) — confirms the gain is from DINO's features being better,
   not just "more features."

Classifier head was also compared on DINO features: Logistic Regression
(~79%), k-NN (75-78%, worse), MLP 256-hidden (81.8%, used in production).

See `ml/README.md` for the full comparison and `experiment_dino.py`/
`experiment_dino2.py` for the scripts that produced these numbers.

## Supported Plants — CHANGED THIS SESSION
`backend/data/plants.json` was updated: **Peepal (Ficus religiosa) and
Banyan (Ficus benghalensis) were replaced with Tulsi (Ocimum tenuiflorum)
and Amla (Phyllanthus emblica)**, keeping the same `plant_id`s (plant_03,
plant_04) so nothing downstream needs to change. Team decision made with
the user. Full current set: plant_01 Indian Laburnum, plant_02 Neem,
plant_03 Tulsi, plant_04 Amla, plant_05 Ashoka Tree, plant_06 Mango,
plant_07 Indian Bael, plant_08 Curry Leaf Tree.

**Why:** Peepal and Banyan are both aerial-rooted fig trees that proved to
be a genuinely hard, structurally similar pair (47-48 mutual
misclassifications, more than either class confused with anything else,
even after dedicated data curation — measured before the duplicate-content
bug was found, but the underlying visual-similarity finding is independent
of that bug). Tulsi and Amla are visually distinct from every other class
and both genuinely native with real conservation/medicinal significance.

**Action needed:** `backend/data/plants.json` facts for Tulsi/Amla were
written from general botanical knowledge (same rigor as the original 8
entries), but still need an independent fact-check pass.

## Completed
- `ml/fetch_dataset.py`: bootstraps/expands `ml/dataset/<plant_id>/` from
  Wikimedia Commons. Iterated many times this session; final, corrected
  version dedupes by content hash against every file already on disk
  (not just this run's history) before saving — see "The duplicate-content
  bug" above for why that matters.
- `ml/dedupe_dataset.py` (new): one-off cleanup that removed the 553
  duplicate files found by hashing the whole dataset.
- Manually reviewed samples from every expansion pass (not just counted
  them). Found and removed, across the whole session: a person eating a
  mango (twice — once was a duplicate re-download of the same junk
  before the dedup fix), a landscape photo with a barely-visible tree, a
  ~35-image cluster of Bodh Gaya temple/pilgrimage photos (technically
  on-topic for the old Peepal class, but useless), a pile of dried seeds
  mislabeled as Neem, a mango leaf misfiled under Ashoka Tree, and an
  unrelated gold-mining signboard misfiled under Curry Leaf Tree.
- `ml/filter_dataset.py`: automated cosine-similarity-to-trusted-centroid
  outlier check. Verified reliable for clearly-wrong content (5/5
  spot-checked flags were genuine junk) but also verified it's blind to
  on-topic-but-non-discriminative photos — used only where confirmed
  reliable; did full manual review for the confusable Peepal/Banyan pair.
- `ml/train.py`: extracts embeddings with 4x augmentation per photo,
  trains the classifier, reports cross-validated accuracy using
  `StratifiedGroupKFold`. Also computes and saves the per-class
  out-of-distribution guard data.
- `ml/inference.py`: exposes `identify_plant(image_bytes) ->
  {"plant_id": str|None, "confidence": float}` — the exact mock contract.
- Out-of-distribution guard, v1 → v2: v1 (similarity to *any* training
  embedding) was built and verified early on a small dataset, then found
  to have silently broken as the dataset grew (8/10 unsupported species
  confidently misidentified in a systematic re-test). Rebuilt as v2
  (similarity to the *predicted class's own* photos, per-class calibrated
  threshold). See "The duplicate-content bug" above for why its measured
  performance changed again after that fix.
- Rewrote `backend/services/plant_ai.py` to call the real model, with an
  automatic fallback to the original mock if ML dependencies or model
  files aren't available.
- Added `torch`, `torchvision` (CPU wheels via `--extra-index-url`),
  `scikit-learn`, `joblib` to `backend/requirements.txt`.
- **Switched the production backbone from MobileNetV3-Small to DINO
  ViT-Small** (self-supervised) after the user pushed for ideas to close
  the remaining gap to 90% and CLIP/ensembling of similar CNNs were ruled
  out (too large / low expected value given the ResNet18 test). Verified
  via honest side-by-side comparison, not assumed — see "Model Approach."
  Pinned to a specific commit of `facebookresearch/dino` (not `main`) for
  reproducibility. Classifier switched to an MLP (from Logistic
  Regression), also compared honestly on the same data.

## Files Changed
- `backend/data/plants.json` (plant_03/plant_04 content replaced —
  Peepal/Banyan → Tulsi/Amla; same IDs, same schema)
- `ml/fetch_dataset.py` (new, iteratively extended; duplicate-content bug
  fixed)
- `ml/dedupe_dataset.py` (new — one-off duplicate-removal cleanup)
- `ml/filter_dataset.py` (new — automated outlier-flagging cleanup pass)
- `ml/train.py` (new, extended with augmentation, grouped CV, the v2
  per-class out-of-distribution guard, and — final change — the DINO
  ViT-Small backbone + MLP classifier)
- `ml/inference.py` (new, rewritten for the v2 guard, then again for DINO)
- `ml/experiment_dino.py`, `ml/experiment_dino2.py` (new — the backbone/
  classifier comparison scripts, kept as documentation)
- `ml/README.md` (full write-up, including the duplicate-content bug, the
  DINO switch, and corrected accuracy/OOD numbers)
- `ml/dataset/plant_01/` … `plant_08/` (61-138 images/class, 837 unique
  photos, after cleanup, expansion, and duplicate removal)
- `ml/model/classifier.joblib`, `classes.json`,
  `class_reference_embeddings.npz`, `class_similarity_thresholds.json`
  (final retrained artifacts, DINO-based)
- `backend/services/plant_ai.py` (rewritten internals only — same
  `identify_plant(image_bytes)` signature/contract; `routes/identify.py`
  untouched)
- `backend/requirements.txt` (added ML dependencies)
- `.gitignore` (added `ml/dataset_rejected/`)

## API Status
`POST /api/identify` unchanged externally. Verified locally with the
backend actually running, via real HTTP requests, after the final
(DINO-based) retrain:
- Real images of all 8 current plants → correctly identified.
- Empty upload / non-image bytes / random noise → handled per contract.
- **10 real photos of confirmed non-project species** (hibiscus, rose,
  sunflower, marigold, bougainvillea, money plant, gulmohar, jasmine,
  aloe, banana): **7/10 correctly rejected** on the final DINO model (up
  from 6/10 on the corrected MobileNetV3 model, and up from an
  originally-measured-but-inflated 9/10 before the duplicate-content bug
  was fixed). Tagetes (marigold), Delonix (gulmohar), and Jasminum
  (jasmine) currently slip through.
- Unknown/malformed requests still handled entirely by
  `routes/identify.py`/`plant_service.py`, untouched by this work.

## Accuracy / Testing
**Honest numbers only — every figure below is from an actual run:**

| Stage | Images/class | Backbone | Accuracy |
|---|---|---|---|
| First pass, no augmentation | ~18-20 | MobileNetV3 | ~62% |
| Cleaned + augmented (later found ~41% duplicate) | 70-217 | MobileNetV3 | 79-87.2% *(inflated, do not use)* |
| After removing 553 duplicate files | 59-126 | MobileNetV3 | ~68.8% |
| After one more careful, dedup-safe expansion | 61-138 | MobileNetV3 | ~71.0% |
| **Switched to DINO ViT-Small + MLP (same data)** | 61-138 | **DINO** | **~81.7-81.8% (final, honest)** |

**The user's target was 90%+. The honest final result is ~81.7-81.8%,
still short of that target but a real ~10.7-10.8 point improvement over
the corrected 71.0% baseline**, achieved by switching to a self-supervised
backbone rather than adding more data (which had shown diminishing
returns under the old backbone). The production run (81.7%) closely
matched the standalone comparison experiment (81.8%), which is a good
sign this isn't a fluke of one particular random CV split.

Per-class F1 and the confusion matrix from the final model are
reproducible via `python ml/train.py`'s output plus a confusion-matrix
script (see `ml/README.md`); not duplicating exact numbers here since
they drift slightly between retrains and this file would go stale.

I did not test on physical/live camera photos of real plants — that
requires either real campus photos or Member 3's camera capture flow.
Member 5: this dataset, these accuracy numbers, and the OOD guard need
your independent QA pass per your test matrix. Please treat both the
~81.7% accuracy and the 7/10 OOD rejection rate as the honest current
baseline, not a target already met.

## Problems
1. **Dataset is still Wikimedia reference images, not real campus
   photos.** This remains the primary ceiling on accuracy.
2. **The duplicate-content bug** — see the dedicated section at the top
   of this file. The single most important thing that happened this
   session: a real, structural bug that inflated every number for
   multiple rounds before being caught by an integrity check I ran on my
   own initiative, not because anything crashed or looked obviously
   wrong. Lesson generalized: when a metric keeps improving suspiciously
   smoothly across many iterations, verify the data pipeline itself, not
   just the latest change.
3. **Broadening fetch queries introduces meaningfully more noise than a
   narrow query** (separate from the duplicate bug) — caught by manually
   reviewing full index ranges, not trusting counts.
4. **Filename-collision data-corruption bug** (separate, smaller, found
   earlier and fixed): a manual deletion left a gap in file numbering; a
   later fetch reused the freed index and silently overwrote a verified
   good image with a re-downloaded duplicate of already-known junk.
   Caught via `git diff --stat`, restored from git history, fixed.
5. **Out-of-distribution guard is real but not strong** — 7/10 on the
   current honest dataset (up from 6/10 after the DINO switch, but still
   not robust). This is a working defense against total failure (the
   original bug: random noise at 87% confidence), not a robust open-set
   detector. A materially better version would need a proper
   negative-example-trained classifier, which needs real project time to
   build and validate.
6. **Deployment size/complexity risk, now larger:** `torch`+`torchvision`
   were already meaningfully bigger than the rest of the backend's
   dependencies; DINO adds a ~85MB runtime download from `github.com` and
   `dl.fbaipublicfiles.com` on first import (pinned to a specific commit
   for reproducibility, but the deploy host still needs outbound access
   to those hosts). Flagged to Member 1 — this needs an actual deploy
   dry run, not an assumption that it'll work.

## Decisions
- Frozen pretrained feature extraction, not fine-tuning: matches dataset
  size, avoids overfitting.
- MobileNetV3-Small over ResNet18 (first comparison): no accuracy
  difference measured between two supervised CNNs.
- DINO ViT-Small over MobileNetV3-Small (second comparison, after the
  user pushed for more ideas toward 90%): real, measured accuracy jump.
  Chose it over CLIP (ruled out as too large) and over ensembling two
  similar supervised CNNs (ruled out by the ResNet18 result as low
  expected value before even testing it).
- MLP classifier over Logistic Regression (on DINO features): measured
  improvement, small enough to not meaningfully change deployment cost.
- Replaced Peepal/Banyan with Tulsi/Amla (team decision with the user).
- Rebuilt the out-of-distribution guard as a per-predicted-class check
  after v1 (pooled similarity) was found broken by testing; recalibrated
  again after both the dedup fix and the DINO switch, rather than
  assuming a threshold computed for one embedding space still applies to
  another.
- When the duplicate-content bug was found, chose to fully deduplicate
  and re-measure rather than keep the higher (known-inflated) numbers —
  reliability and honesty over a better-looking result.
- `plant_ai.py` falls back to the original mock automatically if ML deps/
  model files are missing, rather than crashing the backend.

## Next Tasks
- Re-run the "expand the weakest classes" data-growth cycle *with DINO
  embeddings* — the earlier "diminishing returns" finding was measured
  entirely under the old, weaker backbone; unknown whether more data
  still helps as much now.
- Implement test-time augmentation (TTA) at inference — average
  predictions across several augmented views of one uploaded photo.
  Cheap, not yet tried.
- Consider partial fine-tuning of DINO's last transformer block, or a
  larger DINO variant (ViT-Base) — both discussed with the user as
  higher-effort options if more time is available.
- Coordinate with Member 5 on real campus photos of the 8 (current)
  species to retrain against — still likely the most reliable single fix.
- Coordinate with Member 5 on independently re-testing both the accuracy
  claim and the out-of-distribution guard with their own data/photos.
- Coordinate with Member 1 on a Render/Railway deploy dry run — now more
  important than before, since DINO adds a runtime download from
  `github.com`/`dl.fbaipublicfiles.com` on top of the existing
  torch/torchvision size.
- Coordinate with whoever owns `backend/data/plants.json` on fact-
  checking the new Tulsi/Amla entries.

## Handoff Notes
- **How to start the service:** part of normal backend startup —
  `cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000`.
  No separate process; `plant_ai.py` loads the model once at import time.
- **What endpoint to call:** unchanged — `POST /api/identify`,
  multipart `image` field.
- **What response is returned:** unchanged contract —
  `{plant_id, name, scientific_name, confidence}` or the `error` variant.
- **What model/dependency is required:** `torch`, `torchvision`,
  `scikit-learn`, `joblib` (in `backend/requirements.txt`). Falls back to
  the deterministic mock automatically if these aren't installed or
  `ml/model/*` is missing. **New:** the DINO backbone also needs outbound
  network access to `github.com` and `dl.fbaipublicfiles.com` on first
  import (downloads and caches ~85MB) — not just PyPI/torch's own index.
- **What changed that affects other members:** `backend/data/plants.json`
  plant_03/plant_04 content changed (Peepal/Banyan → Tulsi/Amla).
- **What remains imperfect, stated plainly:** ~81.7-81.8% cross-validated
  accuracy (up from 62% at the start of this session, and up from a
  corrected 71.0% after switching from MobileNetV3 to DINO ViT-Small —
  still short of the 90%+ target). Out-of-distribution guard correctly
  rejects 7/10 confirmed non-project species. Real campus photos remain
  the most likely way to close the rest of the gap; TTA and further
  DINO-based data expansion are the next untried, lower-effort levers.
