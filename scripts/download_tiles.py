# scripts/download_tiles.py
import argparse, csv, os, time
from pathlib import Path
import requests
from urllib.parse import urlparse
from tqdm import tqdm

def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def download(url: str, out_path: Path, retries=3, timeout=60):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # skip if exists and non-empty
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"
    for attempt in range(1, retries+1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(out_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, leave=False, desc=out_path.name
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            # basic sanity
            if out_path.stat().st_size == 0:
                raise IOError("Empty file after download")
            return "ok"
        except Exception as e:
            if attempt == retries:
                return f"error: {e}"
            time.sleep(2 * attempt)

def main():
    ap = argparse.ArgumentParser(description="Download SWISSIMAGE tiles from exported CSV.")
    ap.add_argument("--csv", required=True, help="CSV exported from swisstopo (Exporter tous les liens)")
    ap.add_argument("--out", required=True, help="Output folder, e.g., data/raw/canton_ticino/swissimage")
    ap.add_argument("--max", type=int, default=0, help="Max files to download (0=all)")
    ap.add_argument("--prefix", default="", help="Optional filename prefix, e.g., ticino_")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The CSV format can vary; we try common headers:
    # 'url', 'URL', 'Lien', 'Link'; 'name', 'Name', 'Fichier'
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            url = r.get("url") or r.get("URL") or r.get("Lien") or r.get("Link")
            name = r.get("name") or r.get("Name") or r.get("Fichier") or ""
            if not url:
                continue
            if not name:
                # fallback: filename from URL
                name = Path(urlparse(url).path).name
            rows.append((name, url))

    if args.max > 0:
        rows = rows[:args.max]

    log = []
    for name, url in tqdm(rows, desc="Tiles", unit="file"):
        fname = args.prefix + sanitize(Path(name).stem) + ".tif"
        status = download(url, out_dir / fname)
        log.append((fname, url, status))

    # write simple manifest
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "url", "status"])
        w.writerows(log)

    print(f"Done. Saved {len(log)} entries to {out_dir}")

if __name__ == "__main__":
    main()
