"""
Dataset bootstrap/expansion script (Member 2).

Downloads freely-licensed reference images per curated plant species from
Wikimedia Commons, for prototyping the classifier before real campus
photos are available.

Deduplicates by content hash against every file already on disk for that
plant, not just URLs seen within the current run - a real bug where
separate runs kept re-downloading the same images under new filenames
(553 of 1340 dataset files turned out to be exact duplicates) was found
and fixed here.

Usage:
    python fetch_dataset.py

Writes to: ml/dataset/<plant_id>/<n>.jpg
"""

import hashlib
import json
import pathlib
import time

import requests

HERE = pathlib.Path(__file__).parent
PLANTS_JSON = HERE.parent / "backend" / "data" / "plants.json"
DATASET_DIR = HERE / "dataset"
IMAGES_PER_PLANT = 20
IMAGES_PER_QUERY = 30

HEADERS = {"User-Agent": "NativeFloraAI-Hackathon/1.0 (educational prototype; contact: team)"}
API_URL = "https://commons.wikimedia.org/w/api.php"


def search_commons_images(query: str, limit: int) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,  # File namespace
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": 640,
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    urls = []
    for page in pages.values():
        infos = page.get("imageinfo", [])
        if not infos:
            continue
        info = infos[0]
        mime = info.get("mime", "")
        if not mime.startswith("image/"):
            continue
        url = info.get("thumburl") or info.get("url")
        if url:
            urls.append(url)
    return urls


def fetch_bytes(url: str) -> bytes | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        print(f"    failed: {url} ({exc})")
        return None


def main(target_plant_ids: list[str] | None = None) -> None:
    """
    Expand the dataset with multiple query variants per plant (scientific
    name, common name, and part-specific terms) rather than a single query,
    to pull in more visual variety and a larger pool to pick good images
    from. Pass target_plant_ids to only (re)fetch specific plants.
    """
    plants = json.loads(PLANTS_JSON.read_text(encoding="utf-8"))["plants"]
    for plant in plants:
        plant_id = plant["plant_id"]
        if target_plant_ids and plant_id not in target_plant_ids:
            continue

        scientific_name = plant["scientific_name"]
        common_name = plant["common_name"]
        out_dir = DATASET_DIR / plant_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # Next free numeric index, not just a file count: filenames are not
        # necessarily contiguous (e.g. after a bad image is manually removed),
        # so using the count alone can collide with an existing file and
        # silently overwrite it. Verified this actually happened once.
        existing_files = list(out_dir.glob("*.jpg"))
        existing_indices = [int(f.stem) for f in existing_files if f.stem.isdigit()]
        existing = (max(existing_indices) + 1) if existing_indices else 0

        # Content-hash dedup against every file already on disk, not just
        # this run's search results: separate fetch runs each reset their
        # own "seen this URL" tracking, so the same Commons image (often
        # returned by multiple query variants, or even by the same query in
        # a later run) kept getting re-downloaded under a new filename.
        # Verified this happened extensively - 553 of 1340 files in the
        # dataset turned out to be exact-duplicate content before this fix.
        known_hashes = {
            hashlib.sha256(f.read_bytes()).hexdigest() for f in existing_files
        }

        queries = [
            scientific_name,
            f"{scientific_name} leaf",
            f"{scientific_name} tree",
            f"{scientific_name} flower",
            f"{scientific_name} fruit",
            f"{common_name} plant",
        ]

        seen_urls: set[str] = set()
        all_urls: list[str] = []
        for query in queries:
            print(f"[{plant_id}] searching Commons for '{query}'...")
            try:
                urls = search_commons_images(query, IMAGES_PER_QUERY)
            except Exception as exc:
                print(f"  search failed: {exc}")
                continue
            new = [u for u in urls if u not in seen_urls]
            seen_urls.update(new)
            all_urls.extend(new)
            print(f"  found {len(urls)} ({len(new)} new)")
            time.sleep(0.3)

        print(f"  total candidate pool: {len(all_urls)}")

        saved = existing
        skipped_duplicates = 0
        for url in all_urls:
            content = fetch_bytes(url)
            if content is None:
                time.sleep(0.15)
                continue
            digest = hashlib.sha256(content).hexdigest()
            if digest in known_hashes:
                skipped_duplicates += 1
                time.sleep(0.15)
                continue
            known_hashes.add(digest)
            dest = out_dir / f"{saved:02d}.jpg"
            dest.write_bytes(content)
            saved += 1
            time.sleep(0.15)

        print(f"  saved {saved - existing} new images, skipped {skipped_duplicates} "
              f"duplicates ({saved} total) to {out_dir}")


if __name__ == "__main__":
    main()
