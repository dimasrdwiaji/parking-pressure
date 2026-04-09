import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from collections import defaultdict

# -----------------------------------------------
# Config
# -----------------------------------------------
WMTS_ORIGIN_X    = -285401.92
WMTS_ORIGIN_Y    =  903401.92
RESOLUTION       = 0.21
TILE_SIZE_PX     = 256
TILE_GROUND_SIZE = RESOLUTION * TILE_SIZE_PX

# YOLO-World sets classes via set_classes() before inference.
# Unlike Grounding DINO, there is no period separator — just a list of strings.
# yolov8x-worldv2 is the largest and most accurate variant.
YOLO_MODEL    = "yolov8x-worldv2.pt"
CLASSES       = ["car", "van", "truck", ""]
CONF_THRESHOLD = 0.3   # detection confidence threshold
IOU_THRESHOLD  = 0.5   # NMS IoU threshold

# Only process the first chunk for comparison against SAM3
CHUNK_SIZE  = 5000
BATCH_SIZE  = 8   # YOLO-World is much lighter than SAM3 — larger batch is fine
NUM_WORKERS = 2

IMG_DIR     = "data/imagery/nl/rgb"
OUTPUT_DIR  = "data/imagery/nl/pred"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "detected_cars_yoloworld_chunk1.parquet")


# -----------------------------------------------
# Group tiles into 2x2 blocks (same as car_detection_v3.py)
# -----------------------------------------------
def group_tiles(image_paths):
    """
    NL filename format: GRD_ID_tile_col_tile_row_rel_col_rel_row.jpg
    rsplit from the right gives tile_col at parts[1] and tile_row at parts[2].
    """
    groups = defaultdict(dict)
    for path in image_paths:
        basename  = os.path.basename(path).replace(".jpg", "")
        parts     = basename.rsplit("_", 4)
        col       = int(parts[1])
        row       = int(parts[2])
        group_key = (col // 2, row // 2)
        groups[group_key][(col, row)] = path
    return groups


def stitch_group(group_tiles_dict, group_key):
    """Stitches up to 4 tiles into one 512×512 image."""
    base_col = group_key[0] * 2
    base_row = group_key[1] * 2
    canvas   = np.zeros((TILE_SIZE_PX * 2, TILE_SIZE_PX * 2, 3), dtype=np.uint8)
    for (col, row), path in group_tiles_dict.items():
        px_x = (col - base_col) * TILE_SIZE_PX
        px_y = (row - base_row) * TILE_SIZE_PX
        tile  = np.array(Image.open(path).convert("RGB"))
        canvas[px_y : px_y + TILE_SIZE_PX, px_x : px_x + TILE_SIZE_PX] = tile
    return canvas, base_col, base_row


# -----------------------------------------------
# Dataset
# -----------------------------------------------
class StitchedTileDataset(Dataset):
    def __init__(self, groups):
        self.items = list(groups.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        group_key, tiles_dict = self.items[idx]
        img_array, base_col, base_row = stitch_group(tiles_dict, group_key)
        return img_array, base_col, base_row


# -----------------------------------------------
# Inference
# -----------------------------------------------
def run_yoloworld(groups, model):
    """
    Runs YOLO-World inference on 2x2 stitched tile groups.

    YOLO-World is a single-stage open-vocabulary detector. Unlike the two-stage
    Grounded SAM pipeline (DINO → SAM), it directly outputs bounding boxes in
    one forward pass. This makes it significantly faster.

    Since YOLO-World returns bounding boxes rather than segmentation masks,
    we use the centre of each bounding box as the car centroid rather than
    center_of_mass on a mask. The geographic coordinate conversion is otherwise
    identical to the SAM3 script.

    YOLO-World was trained on a large mix of detection datasets including COCO,
    Objects365, and GoldG. It still suffers from domain shift on nadir aerial
    imagery but tends to outperform Grounding DINO for small overhead vehicles
    because its backbone is optimised for real-time detection efficiency rather
    than semantic grounding, making it less sensitive to viewpoint mismatch.
    """
    dataset    = StitchedTileDataset(groups)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    raw_points = []

    for images, base_cols, base_rows in tqdm(dataloader, desc="  Groups"):
        # Convert batch tensor to list of numpy arrays for YOLO
        images_list = [img.numpy() for img in images]

        # YOLO-World accepts a list of numpy arrays directly.
        # verbose=False suppresses per-image console output.
        results = model.predict(
            images_list,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        for i, result in enumerate(results):
            base_col = base_cols[i].item()
            base_row = base_rows[i].item()

            # Top-left geographic corner of the stitched image
            tile_geo_x = WMTS_ORIGIN_X + (base_col * TILE_GROUND_SIZE)
            tile_geo_y = WMTS_ORIGIN_Y - (base_row * TILE_GROUND_SIZE)

            # result.boxes.xyxy is a tensor of shape (N, 4) in pixel coords
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4): x1, y1, x2, y2

            for x1, y1, x2, y2 in boxes:
                # Centre of bounding box as car centroid
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                geo_x = tile_geo_x + (cx * RESOLUTION)
                geo_y = tile_geo_y - (cy * RESOLUTION)
                raw_points.append([geo_x, geo_y])

    return raw_points


# -----------------------------------------------
# Deduplication
# -----------------------------------------------
def deduplicate(raw_points):
    if not raw_points:
        return pd.DataFrame(columns=["x", "y"])
    df     = pd.DataFrame(raw_points, columns=["x", "y"])
    coords = df[["x", "y"]].values
    labels = DBSCAN(eps=1.5, min_samples=1, n_jobs=-1).fit(coords).labels_
    df["cluster_id"] = labels
    return df.groupby("cluster_id")[["x", "y"]].mean().reset_index(drop=True)


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        print(f"Output already exists at {OUTPUT_FILE}. Delete it to re-run.")
        return

    # Load YOLO-World and set custom classes.
    # set_classes() encodes the class names into the model's text embeddings
    # before inference — this is the "prompt-then-detect" paradigm that makes
    # YOLO-World efficient: text encoding happens once, not per image.
    print(f"Loading YOLO-World ({YOLO_MODEL})...")
    model = YOLO(YOLO_MODEL)
    model.set_classes(CLASSES)
    print(f"Classes set: {CLASSES}")

    # Load tiles and take first chunk only
    all_tiles   = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"Total tiles found: {len(all_tiles):,}")

    all_groups  = group_tiles(all_tiles)
    group_keys  = list(all_groups.keys())[:CHUNK_SIZE]
    first_chunk = {k: all_groups[k] for k in group_keys}
    print(f"Processing first {len(first_chunk):,} groups.")

    # Run inference
    raw_points = run_yoloworld(first_chunk, model)
    print(f"Raw detections: {len(raw_points):,}")

    # Deduplicate
    df_clean = deduplicate(raw_points)
    print(f"After deduplication: {len(df_clean):,}")

    # Save as GeoParquet
    geometry = [Point(xy) for xy in zip(df_clean["x"], df_clean["y"])]
    gdf      = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:28992")
    gdf.to_parquet(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()