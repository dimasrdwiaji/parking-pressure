import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForMaskGeneration,
)
from scipy.ndimage import center_of_mass
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

# Grounding DINO text prompt format: classes separated by periods.
# This is different from SAM3 which takes separate prompt strings.
# All classes in one string is intentional — DINO handles multi-class
# detection in a single forward pass natively.
DINO_TEXT_PROMPT = "small car. parked car. vehicle in parking lot. top view car. van. truck."

# Detection thresholds for Grounding DINO
BOX_THRESHOLD  = 0.3   # confidence threshold for box detection
TEXT_THRESHOLD = 0.3   # text similarity threshold for label assignment

# SAM mask confidence threshold
MASK_THRESHOLD = 0.5

# Only process the first chunk for comparison against SAM3
CHUNK_SIZE = 5000

IMG_DIR    = "data/imagery/nl/rgb"
OUTPUT_DIR = "data/imagery/nl/pred"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "detected_cars_grounded_sam_chunk1.parquet")

# Grounding DINO base is more accurate than tiny, worth the extra VRAM
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# SAM 2 has an improved mask decoder over SAM 1, with better handling of
# ambiguous boundaries between closely packed objects — directly relevant
# for dense parking detection. The HuggingFace API is identical to SAM 1
# so no other code changes are needed.
SAM_MODEL_ID  = "facebook/sam2-hiera-large"


# -----------------------------------------------
# Group tiles into 2x2 blocks (same logic as car_detection_v3.py)
# -----------------------------------------------
def group_tiles(image_paths):
    """
    Groups tile files into 2x2 blocks using tile_col and tile_row.
    NL filename format: GRD_ID_tile_col_tile_row_rel_col_rel_row.jpg
    Uses rsplit to correctly get tile_col and tile_row, not rel_col/rel_row.
    """
    groups = defaultdict(dict)
    for path in image_paths:
        basename = os.path.basename(path).replace(".jpg", "")
        parts    = basename.rsplit("_", 4)
        col      = int(parts[1])
        row      = int(parts[2])
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
def run_grounded_sam(groups, dino_model, dino_processor, sam_model, sam_processor, device):
    """
    Two-stage inference per image:

    Stage 1 — Grounding DINO:
      Takes the image and a text prompt ("car. van. truck.") and returns
      bounding boxes with confidence scores. Multiple classes are detected
      in one forward pass — no confidence competition because DINO scores
      each box independently against each class token.

    Stage 2 — SAM:
      Takes the image and the DINO bounding boxes as spatial prompts and
      returns binary segmentation masks. SAM uses the boxes to constrain
      where it looks, which is why it handles dense/occluded cars better
      than SAM3's free-form instance segmentation — the detector first
      localises each car, then SAM segments it precisely.

    This is fundamentally different from SAM3:
      SAM3: one model, text → masks directly
      Grounded SAM: DINO (text → boxes) → SAM (boxes → masks)
    """
    dataset    = StitchedTileDataset(groups)
    # batch_size=1 because DINO + SAM together use more VRAM than SAM3 alone.
    # DINO-base is ~900MB and SAM-huge is ~2.4GB — they coexist on 16GB
    # but leave limited headroom for activation memory at larger batch sizes.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    raw_points = []

    with torch.no_grad():
        for images, base_cols, base_rows in tqdm(dataloader, desc="  Groups"):
            img_array = images[0].numpy()  # (H, W, 3)
            base_col  = base_cols[0].item()
            base_row  = base_rows[0].item()

            pil_image = Image.fromarray(img_array)

            # ---- Stage 1: Grounding DINO box detection ----
            dino_inputs = dino_processor(
                images=pil_image,
                text=DINO_TEXT_PROMPT,
                return_tensors="pt"
            ).to(device)

            dino_outputs = dino_model(**dino_inputs)

            # post_process_grounded_object_detection handles multi-class text prompts
            # correctly, returning one label string per detected box.
            # target_sizes expects (H, W) tuples.
            # Newer transformers unified box_threshold + text_threshold into
            # a single 'threshold'. input_ids is now read from outputs internally.
            dino_results = dino_processor.post_process_grounded_object_detection(
                dino_outputs,
                threshold=BOX_THRESHOLD,
                target_sizes=[(img_array.shape[0], img_array.shape[1])]
            )[0]

            boxes = dino_results["boxes"]  # shape (N, 4) in xyxy pixel coords

            if len(boxes) == 0:
                continue

            # ---- Stage 2: SAM mask generation from DINO boxes ----
            # SAM expects input_boxes as (batch, num_boxes, 4).
            # We unsqueeze to add the batch dimension.
            sam_inputs = sam_processor(
                images=pil_image,
                input_boxes=boxes.unsqueeze(0).tolist(),
                return_tensors="pt"
            ).to(device)

            sam_outputs = sam_model(**sam_inputs)

            # post_process_masks returns one mask per box.
            # multimask_output=False means SAM returns one mask per prompt
            # instead of 3 candidate masks — simpler and sufficient here.
            # SAM 2's processor does not store reshaped/input sizes —
            # only pixel_values, original_sizes, and input_boxes are returned.
            # We therefore resize predicted masks back to the original image
            # size manually using bilinear interpolation, then threshold them.
            orig_h, orig_w = img_array.shape[0], img_array.shape[1]

            # pred_masks shape: (batch=1, num_boxes, num_mask_candidates, H, W)
            # We take the first (and only) mask candidate per box.
            pred = sam_outputs.pred_masks[0]  # (num_boxes, num_candidates, H, W)
            pred = pred[:, 0, :, :].unsqueeze(1).float()  # (num_boxes, 1, H, W)

            pred_resized = torch.nn.functional.interpolate(
                pred,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )  # (num_boxes, 1, orig_h, orig_w)

            # Apply threshold to get binary masks
            masks = (pred_resized[:, 0, :, :] > MASK_THRESHOLD).cpu().numpy()
            # masks shape: (num_boxes, orig_h, orig_w)

            # ---- Convert masks to geographic centroids ----
            tile_geo_x = WMTS_ORIGIN_X + (base_col * TILE_GROUND_SIZE)
            tile_geo_y = WMTS_ORIGIN_Y - (base_row * TILE_GROUND_SIZE)

            # masks is already a numpy bool array (num_boxes, H, W)
            for mask_np in masks:
                if mask_np.sum() == 0:
                    continue
                cy, cx = center_of_mass(mask_np.astype(np.uint8))
                geo_x  = tile_geo_x + (cx * RESOLUTION)
                geo_y  = tile_geo_y - (cy * RESOLUTION)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Grounding DINO
    print(f"Loading Grounding DINO ({DINO_MODEL_ID})...")
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(
        DINO_MODEL_ID
    ).to(device)
    dino_model.eval()

    # Load SAM
    print(f"Loading SAM ({SAM_MODEL_ID})...")
    sam_processor = AutoProcessor.from_pretrained(SAM_MODEL_ID)
    sam_model     = AutoModelForMaskGeneration.from_pretrained(
        SAM_MODEL_ID
    ).to(device)
    sam_model.eval()

    # Load tiles and take first chunk only
    all_tiles = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"Total tiles found: {len(all_tiles):,}")

    all_groups  = group_tiles(all_tiles)
    group_keys  = list(all_groups.keys())[:CHUNK_SIZE]
    first_chunk = {k: all_groups[k] for k in group_keys}
    print(f"Processing first {len(first_chunk):,} groups ({CHUNK_SIZE} group limit).")

    # Run inference
    raw_points = run_grounded_sam(
        first_chunk, dino_model, dino_processor, sam_model, sam_processor, device
    )

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