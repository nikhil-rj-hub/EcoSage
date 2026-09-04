"""
One-off dataset bootstrap script (Member 2).

Downloads a small set of freely-licensed reference images per curated plant
species from Wikimedia Commons, for prototyping the classifier before real
campus photos are available.

Usage:
    python fetch_dataset.py

Writes to: ml/dataset/<plant_id>/<n>.jpg
"""

import json
import pathlib
import time

import requests

HERE = pathlib.Path(__file__).parent
PLANTS_JSON = HERE.parent / "backend" / "data" / "plants.json"
DATASET_DIR = HERE / "dataset"
IMAGES_PER_PLANT = 20

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


def download(url: str, dest: pathlib.Path) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:
        print(f"    failed: {url} ({exc})")
        return False


def main() -> None:
    plants = json.loads(PLANTS_JSON.read_text(encoding="utf-8"))["plants"]
    for plant in plants:
        plant_id = plant["plant_id"]
        query = plant["scientific_name"]
        out_dir = DATASET_DIR / plant_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{plant_id}] searching Commons for '{query}'...")
        urls = search_commons_images(query, IMAGES_PER_PLANT)
        print(f"  found {len(urls)} candidate images")

        saved = 0
        for i, url in enumerate(urls):
            ext = ".jpg"
            dest = out_dir / f"{saved:02d}{ext}"
            if download(url, dest):
                saved += 1
            time.sleep(0.2)

        print(f"  saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()
