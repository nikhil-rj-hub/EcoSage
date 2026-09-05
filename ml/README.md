# Plant Identification (Member 2)

Real implementation behind `POST /api/identify`. Replaces the mock that
previously lived in `backend/services/plant_ai.py`.

## Approach

Transfer learning via frozen feature extraction (not full fine-tuning):

```
image -> DINO ViT-Small (self-supervised, frozen) -> 384-d embedding
      -> MLP classifier (trained on curated plants)
      -> plant_id + confidence
```

Fine-tuning the backbone itself was ruled out: the curated dataset is
small enough that fine-tuning millions of parameters would badly overfit.
Frozen pretrained features + a classifier on top is the standard, more
data-efficient approach for this dataset size.

**Why DINO, not a supervised CNN (this went through two real iterations,
not a single guess):**
1. Started with MobileNetV3-Small (2.5M params), and compared it
   side-by-side against ResNet18 (11.7M params) - both are *supervised*
   ImageNet classifiers. Result: identical accuracy (62% vs 62% at the
   time). Conclusion: swapping between similarly-trained supervised CNNs
   doesn't help, because they learn correlated features from the same
   labeled-classification objective - the bottleneck wasn't model
   capacity.
2. Later, under pressure to close a large remaining accuracy gap, tried
   DINO ViT-Small (21.7M params) - a *self-supervised* model trained with
   a fundamentally different objective (no labels; learns from an image's
   relationship to augmented views of itself). This is a genuinely
   different signal, not just a bigger version of the same thing. Result:
   a real, honest jump - 71.0% -> 81.8% on the exact same
   cross-validation methodology used throughout this file. Also tried
   concatenating DINO + MobileNetV3 features together, which made things
   *worse* (79-80.5%) - MobileNetV3's weaker signal diluted DINO's
   stronger one rather than adding anything complementary, confirming
   this isn't just "more features = better."
3. Classifier head was also compared honestly: Logistic Regression (~79%
   on DINO features), k-NN (worse, 75-78%), and an MLP (256 hidden units,
   alpha=0.01) at 81.8% - the MLP is what's used in production.

See `experiment_dino.py` and `experiment_dino2.py` for the comparison
scripts (kept as documentation of the methodology, not needed to run the
production pipeline).

## Dataset

`ml/dataset/<plant_id>/*.jpg` — one folder per plant, currently populated
via `fetch_dataset.py`, which pulls freely-licensed reference photos from
Wikimedia Commons using multiple query variants per plant (scientific
name, common name, "leaf", "tree", "flower", "fruit").

**This is a placeholder dataset for prototyping, not real campus photos.**
It went through several rounds of quality control, all done by actually
looking at images, not just trusting search results or an automated score:

1. An initial small batch (~18-20 images/class) was manually spot-checked;
   one clearly wrong image was found and removed (a photo of a person
   eating a mango, returned by a Commons search for "Mangifera indica").
2. Broadening the queries to grow the dataset (70-130 images/class)
   introduced a much higher noise rate — a landscape photo with a
   barely-visible distant tree, a Bodh Gaya temple photo where the
   original Peepal was mostly hidden behind ironwork, a pile of dried
   seeds mislabeled as Neem, a mango leaf misfiled under Ashoka Tree, and
   an unrelated gold-mining signboard misfiled under Curry Leaf Tree. An
   automated cosine-similarity-to-trusted-centroid filter
   (`filter_dataset.py`) was tried to scale this up, but was verified to
   miss exactly this kind of error (a landscape shot scored 0.62-0.70
   similarity - well above the cutoff used at the time - because "green
   foliage against sky" reads as generically similar regardless of
   whether the target plant is actually the subject). It *is* reliable
   for clearly-wrong content, just not for on-topic-but-non-discriminative
   photos - those need manual review.
3. **Peepal and Banyan were later replaced with Tulsi and Amla** (team
   decision - see below), and every subsequent expansion round for every
   class was manually reviewed the same way, at the "spot-check the full
   index range" level of rigor, not just a handful of samples.
4. **A much bigger, structural bug was found afterward, by hashing every
   file in the dataset:** repeated `fetch_dataset.py` runs each reset
   their own within-run duplicate tracking, so the same Commons photo
   kept getting downloaded and re-added under a new filename across
   different runs. **553 of 1340 dataset files (41%) turned out to be
   exact-duplicate content.** This wasn't just wasted space - it silently
   defeated `train.py`'s `StratifiedGroupKFold` leak-prevention, which
   only protects against *augmented copies* of one file leaking across
   CV folds; two different files with identical raw content still get
   different group IDs and can land in different folds, letting the
   model get cross-validated against near-duplicates of its own training
   data. This inflated every accuracy number reported earlier in this
   file's history - **see "Known accuracy" for the corrected numbers.**
   Fixed by (a) `dedupe_dataset.py`, a one-off cleanup that removed the
   553 duplicates, and (b) fixing `fetch_dataset.py` to hash against
   every file already on disk before saving a new one, not just URLs
   seen in the current run.

**Species swap:** `plant_03`/`plant_04` originally held Peepal (Ficus
religiosa) and Banyan (Ficus benghalensis) - both aerial-rooted fig trees
that proved to be a genuinely hard, structurally similar pair (47-48
mutual misclassifications, more than either class confused with anything
else, even after dedicated cleanup). The team decided to swap them for
**Tulsi (Ocimum tenuiflorum)** and **Amla (Phyllanthus emblica)** instead -
both genuinely native, common on Indian campuses, and visually distinct
from every other class (small aromatic herb vs. feathery pinnate-leaved
tree). `backend/data/plants.json` was updated accordingly with real,
verifiable botanical facts (not fabricated) - still worth an independent
fact-check pass by whoever owns that data.

Final counts, after cleanup/expansion rounds *and* the duplicate-removal
fix: 61-138 images/class (837 unique source photos) across all 8 plants.
Still not real campus photos of the team's actual specimens - see "Known
accuracy" for why that remains the highest-leverage improvement still on
the table.

## Known accuracy (measured, not assumed - and corrected once already)

Grouped + stratified cross-validation (each source photo's augmented
copies are kept in the same fold, so augmentation can't leak across
train/validation and inflate the number - see `train.py`).

**Every number below labeled "before dedup fix" was measured on a dataset
later found to be ~41% duplicate content and is known to be inflated -
kept here for the record, not as something to quote.** The `after dedup
fix` numbers are the honest ones.

| Stage | Images/class | Backbone + classifier | Accuracy |
|---|---|---|---|
| First pass, no augmentation | ~18-20 | MobileNetV3 + LogReg | ~62% |
| Cleaned + augmented | 70-117 | MobileNetV3 + LogReg | ~79% (before dedup fix) |
| Peepal/Banyan -> Tulsi/Amla swap | 70-159 | MobileNetV3 + LogReg | ~81% (before dedup fix) |
| Expanded weakest classes (round 1) | 70-197 | MobileNetV3 + LogReg | ~86.7% (before dedup fix) |
| Expanded weakest classes (round 2) | 121-218 | MobileNetV3 + LogReg | ~87.2% (before dedup fix) |
| After removing 553 duplicate files | 59-126 | MobileNetV3 + LogReg | ~68.8% |
| After one more careful (dedup-safe) expansion | 61-138 | MobileNetV3 + LogReg | ~71.0% |
| **Switched backbone to DINO ViT-Small + MLP** | 61-138 (same data) | **DINO + MLP** | **~81.7-81.8% (current, honest)** |

**The user's target was 90%+; this was not reached.** The single biggest
move this session was switching from a supervised CNN (MobileNetV3) to a
self-supervised one (DINO ViT-Small) on the *same, already-deduplicated*
dataset - a real, direct +10.7-10.8 point jump (71.0% -> 81.7-81.8%,
both cross-validated the same way, and the production run's 81.7%
matched the standalone experiment's 81.8% closely, which is a good sign
it's not a fluke of one random split). This is a materially different
result from just adding more data, which had been showing diminishing
returns under the old backbone.

**To get closer to 90%+, in order of expected value (updated after the
DINO switch):**
1. Re-run the "expand the weakest classes" data-growth cycle *with DINO
   embeddings* - the earlier "diminishing returns" conclusion was
   measured entirely under the old, weaker backbone; it's not yet known
   whether more data still helps as much now that the underlying
   representation is stronger.
2. Test-time augmentation (TTA) at inference - average predictions
   across several augmented views of the same uploaded photo. Cheap,
   stacks with anything else, not yet implemented.
3. Partial fine-tuning of DINO's last transformer block (not the whole
   backbone) - a middle ground between "fully frozen" (current) and
   "fully fine-tuned" (ruled out as too likely to overfit on this much
   data). Untested.
4. A larger DINO variant (ViT-Base instead of ViT-Small) - self-supervised
   representation quality tends to scale with model size, the same lever
   that got MobileNetV3 -> DINO ViT-Small working. Bigger deployment
   footprint (~330MB vs ~85MB), so this needs the Member 1 conversation
   below to go further, not less, before committing to it.
5. Real campus photos of the team's actual specimens - still true, still
   likely the most reliable single fix, still not something produced this
   session.

Do not quote a higher number than what `train.py` prints for the current
data in the presentation. If in doubt, rerun `python train.py` and read
the number it prints - do not trust a number written in a document,
including this one.

## Out-of-distribution guard

The master prompt and the QA test plan both require the system to say "I
don't know" rather than confidently guess when shown something outside
the curated 8 species (or a non-plant image). This guard was built,
broke silently as the dataset grew, and was caught and fixed by actually
testing it against real out-of-set photos - not by inspection or
assumption. Worth reading in full since it's a real lesson about this
kind of check at scale:

**v1** (rejects if the query isn't similar to *any* training embedding,
checked against the whole pool across all classes): built and verified
early in the session against random noise (correctly rejected) and one
photo of an unsupported species, a rose (correctly rejected). This felt
sufficient at the time.

**v1 broke as the dataset grew.** Once the dataset scaled up to ~1300
source photos (~6700 with augmentation) across visually diverse queries
("leaf", "tree", "flower", "fruit" for 8 species), a systematic test
against 10 confirmed non-project species (hibiscus, rose, sunflower,
marigold, bougainvillea, money plant, gulmohar, jasmine, aloe, banana)
found that **8 of 10 were confidently misidentified** as one of our 8
plants, several at 86-100% confidence. Root cause: with thousands of
diverse reference embeddings spanning many leaf/flower/fruit shapes and
colors, almost *any* photo of *any* plant scores high similarity to
*something* in that pool by chance - the check had become nearly
meaningless. This was only found because the user specifically asked for
this exact test case to be verified, not assumed.

**v2** (current): instead of "similar to anything we've ever seen,"
require the query to be similar to photos of *the specific class the
classifier predicted* - using a threshold calibrated per class from that
class's own data (the 5th percentile of how similar that class's own
photos are to each other, computed from original non-augmented photos
only, since augmented near-duplicates of the same photo would make
same-class similarity look artificially higher than it really is).
Implemented in `train.py` (writes
`model/class_reference_embeddings.npz` and
`model/class_similarity_thresholds.json`) and `inference.py`.

**Verified immediately after the v2 fix, on the (still duplicate-inflated)
dataset at the time:** 9 of 10 out-of-set test photos correctly rejected.

**That 9/10 number did not hold up.** After the duplicate-content bug
(see "Dataset" above) was found and the dataset was cleaned to its honest
size, the per-class thresholds were recalibrated from the smaller, clean
data and **the same 10-photo test dropped to 6 of 10 correctly rejected**
(Rosa, Sunflower, Marigold, and Banana all slipped through on the
corrected model; Hibiscus, Bougainvillea, Money Plant, Gulmohar, Jasmine,
and Aloe are still correctly rejected). This makes sense in hindsight:
the per-class similarity thresholds are calibrated from each class's own
photo set, and a smaller, less redundant photo set gives a less precise
estimate of "how similar do this species' own photos look to each
other" - the guard is real and does something, but it is weaker on this
corrected, smaller dataset than the (misleading) 9/10 result suggested.

**6/10 on this specific 10-photo test set, on the MobileNetV3 model.**

**Re-tested again after switching the backbone to DINO ViT-Small (see
"Approach" above) and recalibrating the per-class thresholds on DINO
embeddings: 7/10 correctly rejected** (Hibiscus, Rosa, Helianthus,
Bougainvillea, Epipremnum, Aloe, Musa now correctly rejected; Tagetes,
Delonix, and Jasminum still slip through, misidentified as Ashoka/
Bael/Neem respectively). A small improvement, consistent with DINO's
features being more discriminative generally - but not dramatically
better, and still not a strong guarantee.

**This rate has changed three times now as the model changed (9/10 on
duplicate-inflated data -> 6/10 after fixing that -> 7/10 after switching
to DINO) - it is clearly sensitive to exactly which model and dataset are
in use.** Whoever retrains this model should re-run this same kind of
test rather than trust any single number, including this one.

**For Member 5:** please retest this specifically as part of your QA pass
(the "unsupported plant" test case in your prompt) with your own set of
non-project-species photos, and treat 7/10 as the honest current baseline
to compare against, not a target already met.

## Files

- `fetch_dataset.py` — bootstraps/expands `dataset/` from Wikimedia Commons
  using multiple query variants per plant. Re-run (optionally with a
  `target_plant_ids` list) if a class needs more images - but manually
  review what it pulls in before training on it (see "Dataset" above).
  Dedupes by content hash against every file already on disk (fixed after
  finding 553 duplicate files - see "Dataset").
- `dedupe_dataset.py` — one-off cleanup that removed the 553 duplicate
  files found by hashing the dataset. Shouldn't be needed again now that
  `fetch_dataset.py` dedupes on save, but safe to re-run (`--apply`) as a
  sanity check if the dataset is ever suspected of drifting again.
- `filter_dataset.py` — automated cosine-similarity outlier check against
  each class's originally-verified images. Useful for catching clearly
  wrong content at scale, but *not* a substitute for manually reviewing
  visually-similar/confusable classes - see "Dataset" above for what it
  misses and why. `--apply` moves flagged images to `dataset_rejected/`
  instead of deleting them outright.
- `train.py` — extracts DINO ViT-Small embeddings (with augmentation),
  trains the MLP classifier, prints cross-validated accuracy, saves
  `model/classifier.joblib`, `model/classes.json`,
  `model/class_reference_embeddings.npz`, and
  `model/class_similarity_thresholds.json` (the last two are the
  out-of-distribution guard's data - see above).
- `inference.py` — loads the trained artifacts and exposes
  `identify_plant(image_bytes) -> {"plant_id": str|None, "confidence": float}`,
  the same contract the mock it replaces used.
- `model/` — trained artifacts (regenerate by running `train.py`; not
  meaningfully useful without `dataset/`, but small enough to commit so the
  backend works without a retrain step).
- `experiment_dino.py`, `experiment_dino2.py` — the side-by-side
  backbone/classifier comparison scripts that led to switching to DINO
  (see "Approach" above). Not part of the production pipeline; kept as
  documentation of the methodology, not something you need to run.

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

**New since switching to DINO:** the DINO ViT-Small weights (~85MB) are
downloaded at runtime via `torch.hub.load(...)` the first time
`plant_ai.py` imports successfully, from `github.com` and
`dl.fbaipublicfiles.com` (pinned to a specific commit, not "main" - see
the comment in `train.py`/`inference.py` - so it won't silently change,
but the deploy host still needs outbound access to those two domains on
first run, in addition to whatever `torchvision` already needed for
MobileNet's weights). If the deploy environment blocks outbound requests
to unfamiliar hosts, this will fail - the fallback mock still protects
against a hard crash, but confirm this actually reaches those hosts in a
deploy dry run rather than assuming it will.
