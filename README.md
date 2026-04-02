# Parking Pressure Estimation for Pan-European Scale

Estimating parking pressure across Europe using aerial orthoimagery and OpenStreetMap data. The pipeline detects parked cars from high-resolution aerial photos, clusters them into parking areas, and derives a parking pressure metric from parking occupancy rate in a 500m grid.

> **Note:** Work in progress.

---

## Research context

Parking pressure, the mismatch between parking supply and vehicle use, impacts urban mobility and public health. When parking is scarce, drivers spend extra time cruising, increasing traffic. However, when oversupplied, it encourages higher car ownership and use. Both contribute to pollution exposure, sedentary lifestyles, and inefficient use of urban space, with broader economic impacts in European cities.

Existing studies are limited in scale and lack standardized methods, while EU policy (e.g., Sustainable Urban Mobility Plans) calls for reducing parking supply at a continental level. This project addresses the absence of EU-wide parking data by detecting and mapping parked cars using deep learning. CNN-based methods perform well locally, but poor transferability to other countries with different spatial resolution, spectral information, and temporal acquisition. Therefore, this project use of foundation models like Segment Anything (SAM), which enable zero/shot segmentation.

Where parking area is currently derived from a rotated bounding box fitted to each car cluster. 

---

## Pipeline overview

```
1. Download aerial imagery
        ↓
2. Car detection (SAM3)
        ↓
3. Filter road cars (OSM road buffer + spatial join)
        ↓
4. Cluster parked cars (DBSCAN)
        ↓
5. Estimate parking area (rotated bounding box)
        ↓
6. Compute occupancy rate
        ↓
7. Aggregate to 500m grid
```

---

## Scripts — run in this order

### Step 1 — Download aerial imagery

**`scripts/download_wmts_nl.py`**
Downloads aerial orthoimagery tiles from the Dutch PDOK WMTS service (25cm resolution, EPSG:28992, zoom level 14). Tiles are saved as individual JPEG files named `{GRD_ID}_{tile_col}_{tile_row}.jpg`. Includes a resume mechanism via a cached task parquet file.

**`scripts/download_wmts_fr.py`**
Same as above for France, using the IGN Géoplateforme WMTS service. Layer: `HR.ORTHOIMAGERY.ORTHOPHOTOS.L93`, TileMatrixSet: `2154_10cm_10_20`, zoom level 20 (20cm resolution, EPSG:2154). Tiles are named `{tile_col}_{tile_row}.jpg`. Includes duplicate-tile prevention via a `seen_tiles` set and resume via filtered task parquet.

NOTE: WILL BE STANDARDIZED LATER.

---

### Step 2 — Car detection

**`scripts/car_detection_v3.py`**
Runs SAM3 (`facebook/sam3`) inference on aerial tiles to detect parked cars. Key design decisions:

1. **2×2 tile stitching:** adjacent tiles are stitched into 512×512 images before inference. This reduces the number of vision encoder forward passes by ~4× compared to processing individual 256×256 tiles.
2. **Multi-prompt inference:** three prompts ("car", "van", "truck") are run per image. The vision encoder runs once per image batch. The decoder runs once per prompt reusing cached vision features.
3. **Chunked processing with resume:** tiles are processed in chunks of 5000 groups. After each chunk, detections are saved to `data/imagery/{country}/pred/chunks/chunk_XXXXX.parquet` and completed tile paths are appended to `progress.txt`. On restart, completed tiles are skipped.
4. **Two-stage deduplication:** DBSCAN deduplication runs per chunk (removes within-tile boundary duplicates and cross-prompt duplicates), then once more on the merged output (removes cross-chunk boundary duplicates).

Output: Geoparquet data of car clusters.

---

### Step 3–6 — Parking pressure estimation

`scripts/pressure_estimation_v2.py`
Takes detected car centroids and OSM road data as inputs. Runs pressure estimation pipeline:

1. Road centerline extraction.

2. Road buffering.

3. Road car filtering of raw detected cars with OSM road layer to remove cars actively driven.

4. Car clustering.

5. Cluster geometry and estimate occupancy rate

Output: GeoParquet of cluster polygons
---

## Data

| Dataset | Source | Format | Notes |
|---|---|---|---|
| NL aerial imagery | PDOK WMTS | JPEG tiles | ~188GB for full NL |
| FR aerial imagery | IGN Géoplateforme | JPEG tiles | HR BD Ortho 20cm |
| Global human settlement layer | JRC | Parquet | Used to pre-filter tiles to settled areas |
| EUROSTAT 500m grid | EUROSTAT | GeoParquet | `data/grid_500m_filtered.parquet` |
| OSM raw PBF | Geofabrik | PBF | Per-country full extract |
| OSM filtered (parking) | Derived | PBF | Output of existing parking extraction pipeline |

---

## To-do list
1. Standardize image request, particularly images delivered via WMTS. Script would run multiple instance, each will download data for a country.

2. Try other detector model. SAM3 still miss cars, likely due to image size. Try DINO or YOLO world (quick test on playground shows that YOLO, while faster, perform poorly compared to SAM3).

3. Estimate parking area via OSM residual. Get OSM data per grid, remove non-parking footprint (buildings, vegetation, water body, buffered roads, etc). The residual becomes the estimated parking area.