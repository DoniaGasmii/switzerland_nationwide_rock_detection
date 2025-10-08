# How to use the scripts 


### download_tiles.py 

**How to use (e.g Geneva):**

1. On the SWISSIMAGE page:

   * Mode de sélection → **Sélection par canton** → **Ticino**
   * Format: **Cloud Optimized GeoTIFF**
   * Résolution: **0.1 m**
   * Système de coordonnées: **MN95 (LV95 / EPSG:2056)**
   * Click **Chercher**, then **Exporter tous les liens** → save the CSV to `data/URLs/canton_geneva`.
2. Run:

```bash
python scripts/download_tiles.py \
  --csv data/URLs/canton_geneva/ch.swisstopo.swissimage-dop10-eaNJ5jko.csv \
  --out data/raw/canton_geneva/swissimage \
  --prefix geneva_
```

This gives you the **native 1×1 km COG tiles** for Geneva, ready for your next steps.