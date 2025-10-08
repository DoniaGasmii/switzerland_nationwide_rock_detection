#!/usr/bin/env python3
"""
Download SWISSIMAGE tiles (or any GeoTIFFs) from a Swisstopo CSV export
or from a plain text file of URLs.
Works with or without headers.
"""

import argparse, csv, os, re, time
from pathlib import Path
import requests
from urllib.parse import urlparse
from tqdm import tqdm

# -----------------------------
# Helpers
# -----------------------------
def sanitize(name: str) -> str:
    """Make filenames safe"""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")

def extract_urls_from_csv(csv_path):
    """Try to extract (name, url) pairs from CSV — works even without headers."""
    rows = []
    http_re = re.compile(r"https?://\S+")
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        # try sniff delimiter
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
            reader = csv.reader(f, dialect)
        except Exception:
            f.seek(0)
            reader = csv.reader(f, delimiter=";")

        for line in reader:
            joined = " ".join(line)
            m = http_re.search(joined)
            if not m:
                continue
            url = m.group(0).strip().replace("\\", "")
            # try to extract a name (if any token looks like a .tif filename)
            name = None
            for token in line:
                if token.lower().endswith(".tif"):
                    name = Path(token.strip()).name
                    break
            if not name:
                name = Path(urlparse(url).path).name
            rows.append((name, url))
    # remove duplicates
    uniq = {}
    for n, u in rows:
        uniq[u] = n
    return [(v, k) for k, v in uniq.items()]

def download(url: str, out_path: Path, retries=3, timeout=120):
    """Stream download with progress bar and retry"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(out_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, leave=False, desc=out_path.name
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            if out_path.stat().st_size == 0:
                raise IOError("Empty file after download")
            return "ok"
        except Exception as e:
            if attempt == retries:
                return f"error: {e}"
            time.sleep(2 * attempt)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Download SWISSIMAGE tiles or URLs from CSV (with or without headers).")
    ap.add_argument("--csv", required=True, help="CSV or text file containing URLs (one per line or with metadata).")
    ap.add_argument("--out", required=True, help="Output folder, e.g., data/raw/canton_ticino/swissimage")
    ap.add_argument("--max", type=int, default=0, help="Max number of files to download (0=all)")
    ap.add_argument("--prefix", default="", help="Optional filename prefix, e.g., ticino_")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Reading links from {args.csv} ...")
    rows = extract_urls_from_csv(args.csv)
    print(f"✅ Found {len(rows)} URLs.")

    if args.max > 0:
        rows = rows[:args.max]

    log = []
    for name, url in tqdm(rows, desc="Downloading tiles", unit="file"):
        fname = args.prefix + sanitize(Path(name).stem) + ".tif"
        status = download(url, out_dir / fname)
        log.append((fname, url, status))

    # Save manifest
    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "url", "status"])
        writer.writerows(log)
    print(f"✅ Done. Downloaded {len(log)} files → {manifest}")

if __name__ == "__main__":
    main()
