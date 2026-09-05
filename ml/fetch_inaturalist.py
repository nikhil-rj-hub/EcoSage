"""
Dataset expansion from iNaturalist (Member 2).

A second, different image source from Wikimedia Commons (see
fetch_dataset.py), tried after Commons' query pool for these 8 species
was confirmed exhausted (every candidate in a full 6-query pass came back
as an exact duplicate already on disk) and Wikimedia's API began
rate-limiting with an explicit "contact noc@wikimedia.org" warning.

iNaturalist observations are community-identified ("research grade" =
verified by multiple people), real field photos taken by observers -
qualitatively different from Commons' mix of studio/product/temple/press
photos, and a much larger pool (thousands of research-grade observations
per species for these common Indian plants).

License filter matches common permissive/attribution licenses (CC0,
CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA) - consistent with a hackathon
prototype's non-commercial, educational use; flag to the team before any
commercial/production use given some of these are NC (non-commercial)
licenses.

Usage:
    python fetch_inaturalist.py

Writes to: ml/dataset/<plant_id>/<n>.jpg (continuing existing numbering,
deduped by content hash the same way fetch_dataset.py is).
"""

import hashlib
import json
import pathlib
import time

import requests

HERE = pathlib.Path(__file__).parent
PLANTS_JSON = HERE.parent / "backend" / "data" / "plants.json"
DATASET_DIR = HERE / "dataset"

API_URL = "https://api.inaturalist.org/v1/observations"
HEADERS = {"User-Agent": "NativeFloraAI-Hackathon/1.0 (educational prototype; contact: team)"}
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"
PER_PAGE = 50
MAX_PAGES = 2  # up to 100 observations per species, more than enough given multiple photos each
TARGET_NEW_PER_CLASS = 40


def search_observations(scientific_name: str, page: int) -> list[dict]:
    params = {
        "taxon_name": scientific_name,
        "photos": "true",
        "quality_grade": "research",
        "photo_license": LICENSES,
        "per_page": PER_PAGE,
        "page": page,
        "order_by": "votes",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def photo_url(photo: dict, size: str = "medium") -> str:
    # iNaturalist photo URLs end in a size token (square/small/medium/large/original).
    url = photo["url"]
    for token in ("square", "small", "medium", "large", "original"):
        if url.endswith(f"{token}.jpg") or url.endswith(f"{token}.jpeg"):
            return url.rsplit(f"{token}.", 1)[0] + f"{size}.jpg"
    return url


def fetch_bytes(url: str) -> bytes | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        print(f"    failed: {url} ({exc})")
        return None


def main(target_plant_ids: list[str] | None = None) -> None:
    plants = json.loads(PLANTS_JSON.read_text(encoding="utf-8"))["plants"]
    for plant in plants:
        plant_id = plant["plant_id"]
        if target_plant_ids and plant_id not in target_plant_ids:
            continue

        scientific_name = plant["scientific_name"]
        out_dir = DATASET_DIR / plant_id
        out_dir.mkdir(parents=True, exist_ok=True)

        existing_files = list(out_dir.glob("*.jpg"))
        existing_indices = [int(f.stem) for f in existing_files if f.stem.isdigit()]
        saved = (max(existing_indices) + 1) if existing_indices else 0
        known_hashes = {
            hashlib.sha256(f.read_bytes()).hexdigest() for f in existing_files
        }

        print(f"[{plant_id}] searching iNaturalist for '{scientific_name}'...")
        photo_urls = []
        for page in range(1, MAX_PAGES + 1):
            try:
                observations = search_observations(scientific_name, page)
            except Exception as exc:
                print(f"  search failed: {exc}")
                break
            if not observations:
                break
            for obs in observations:
                for photo in obs.get("photos", []):
                    photo_urls.append(photo_url(photo))
            time.sleep(1.0)  # be polite to iNaturalist's API

        print(f"  found {len(photo_urls)} candidate photos across observations")

        new_saved = 0
        for url in photo_urls:
            if new_saved >= TARGET_NEW_PER_CLASS:
                break
            content = fetch_bytes(url)
            if content is None:
                time.sleep(0.3)
                continue
            digest = hashlib.sha256(content).hexdigest()
            if digest in known_hashes:
                time.sleep(0.3)
                continue
            known_hashes.add(digest)
            dest = out_dir / f"{saved:02d}.jpg"
            dest.write_bytes(content)
            saved += 1
            new_saved += 1
            time.sleep(0.3)

        print(f"  saved {new_saved} new images ({saved} total) to {out_dir}")


if __name__ == "__main__":
    main()
