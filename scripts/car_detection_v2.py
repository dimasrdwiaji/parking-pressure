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
from transformers import Sam3Processor, Sam3Model
from scipy.ndimage import center_of_mass
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# -----------------------------------------------
# Config
# -----------------------------------------------
WMTS_ORIGIN_X    = -285401.92
WMTS_ORIGIN_Y    =  903401.92
RESOLUTION       = 0.21
TILE_SIZE_PX     = 256
TILE_GROUND_SIZE = RESOLUTION * TILE_SIZE_PX

# Text prompts to run per image.
# Each prompt runs separately against cached vision features — no confidence competition.
PROMPTS = ["car", "van", "truck"]

# How many tiles to process in one chunk.
# After each chunk: deduplicate and save. This limits data loss if the job crashes.
CHUNK_SIZE = 5000

CONFIDENCE_THRESHOLD = 0.5

IMG_DIR        = "data/imagery/nl/rgb"
OUTPUT_DIR     = "data/imagery/nl/pred"
CHUNKS_DIR     = os.path.join(OUTPUT_DIR, "chunks")   # one parquet per chunk
PROGRESS_FILE  = os.path.join(OUTPUT_DIR, "progress.txt")  # stores completed tile paths
FINAL_OUTPUT   = os.path.join(OUTPUT_DIR, "detected_cars_final.parquet")


# -----------------------------------------------
# Dataset
# -----------------------------------------------
class WMTSTileDataset(Dataset):
    """Loads JPEGs and parses tile col/row from the filename."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        basename = os.path.basename(path).replace(".jpg", "")
        parts = basename.rsplit("_", 4)
        col = int(parts[1])
        row = int(parts[2])
        img_array = np.array(Image.open(path).convert("RGB"))
        return img_array, col, row, path


# -----------------------------------------------
# Per-chunk inference
# -----------------------------------------------
def run_chunk(image_paths, model, processor, device):
    """
    Processes one chunk of tiles. For each image:
      1. Encode the image ONCE with get_vision_features()
      2. Loop over each prompt, reusing the cached vision features
      3. Collect all centroids from all prompts
    Then deduplicate within the chunk and return a clean DataFrame.

    Why encode once and loop prompts?
    get_vision_features() runs the heavy vision transformer (450M params).
    The prompt decoder is much lighter. By caching vision features we run
    the expensive part once per image regardless of how many prompts we use.
    """
    dataset    = WMTSTileDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    raw_points = []  # list of [geo_x, geo_y]

    with torch.no_grad():
        for images, cols, rows, paths in tqdm(dataloader, desc="  Tiles", leave=False):
            images_list = [img.numpy() for img in images]

            # --- Step 1: encode all images in the batch ONCE ---
            # processor called with images only (no text) gives us pixel_values
            img_inputs = processor(images=images_list, return_tensors="pt").to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vision_embeds = model.get_vision_features(
                    pixel_values=img_inputs.pixel_values
                )

            # --- Step 2: loop over prompts, reusing vision_embeds ---
            for prompt in PROMPTS:
                # processor called with text only gives us text token tensors
                text_inputs = processor(
                    text=[prompt] * len(images_list),
                    return_tensors="pt"
                ).to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        **text_inputs,
                        vision_embeds=vision_embeds   # reuse cached image encoding
                    )

                target_sizes = [(img.shape[0], img.shape[1]) for img in images_list]
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=CONFIDENCE_THRESHOLD,
                    target_sizes=target_sizes
                )

                # --- Step 3: convert masks to geo coordinates ---
                for i in range(len(images_list)):
                    col = cols[i].item()
                    row = rows[i].item()

                    tile_geo_x = WMTS_ORIGIN_X + (col * TILE_GROUND_SIZE)
                    tile_geo_y = WMTS_ORIGIN_Y - (row * TILE_GROUND_SIZE)

                    masks_np = results[i]["masks"].cpu().numpy()
                    for mask in masks_np:
                        if mask.sum() == 0:
                            continue
                        cy, cx = center_of_mass(mask.astype(np.uint8))
                        geo_x = tile_geo_x + (cx * RESOLUTION)
                        geo_y = tile_geo_y - (cy * RESOLUTION)
                        raw_points.append([geo_x, geo_y])

    if not raw_points:
        return pd.DataFrame(columns=["x", "y"])

    # --- Per-chunk deduplication ---
    # This removes detections of the same car from:
    #   - tile border overlaps (same car seen in two adjacent tiles)
    #   - multiple prompts detecting the same vehicle
    df = pd.DataFrame(raw_points, columns=["x", "y"])
    df = deduplicate(df, tolerance_meters=1.5)
    return df


# -----------------------------------------------
# Deduplication (shared by per-chunk and final)
# -----------------------------------------------
def deduplicate(df, tolerance_meters=1.5):
    """
    Clusters points within tolerance_meters of each other into one detection.
    min_samples=1 means a lone point is its own cluster (not treated as noise).
    """
    if df.empty:
        return df
    coords     = df[["x", "y"]].values
    clustering = DBSCAN(eps=tolerance_meters, min_samples=1, n_jobs=-1).fit(coords)
    df         = df.copy()
    df["cluster_id"] = clustering.labels_
    return df.groupby("cluster_id")[["x", "y"]].mean().reset_index(drop=True)


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM3 on {device}...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model     = Sam3Model.from_pretrained(
        "facebook/sam3", torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # --- Load all tile paths ---
    all_tiles = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"Total tiles found: {len(all_tiles):,}")

    # --- Resume: skip tiles already processed ---
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            done = set(line.strip() for line in f if line.strip())
        remaining = [p for p in all_tiles if p not in done]
        print(f"Resuming — {len(done):,} tiles already done, {len(remaining):,} remaining.")
    else:
        done      = set()
        remaining = all_tiles

    if not remaining:
        print("All tiles already processed. Skipping to final merge.")
    else:
        # --- Split remaining tiles into chunks ---
        chunks = [
            remaining[i : i + CHUNK_SIZE]
            for i in range(0, len(remaining), CHUNK_SIZE)
        ]
        print(f"Processing {len(chunks)} chunks of up to {CHUNK_SIZE} tiles each.")

        for chunk_idx, chunk_paths in enumerate(chunks):
            chunk_num = len(os.listdir(CHUNKS_DIR))  # next chunk file number
            chunk_file = os.path.join(CHUNKS_DIR, f"chunk_{chunk_num:05d}.parquet")

            print(f"\nChunk {chunk_idx + 1}/{len(chunks)} — {len(chunk_paths)} tiles")

            chunk_df = run_chunk(chunk_paths, model, processor, device)

            # Save chunk result even if empty (keeps chunk numbering consistent)
            chunk_df.to_parquet(chunk_file, index=False)
            print(f"  Saved {len(chunk_df):,} detections → {chunk_file}")

            # Mark these tiles as done
            with open(PROGRESS_FILE, "a") as f:
                for p in chunk_paths:
                    f.write(p + "\n")

    # --- Final merge and cross-chunk deduplication ---
    print("\nMerging all chunks...")
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "chunk_*.parquet")))

    if not chunk_files:
        print("No chunk files found. Nothing to merge.")
        return

    all_dfs = [pd.read_parquet(cf) for cf in chunk_files]
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total points before final dedup: {len(combined):,}")

    # Final deduplication handles only cross-chunk boundary duplicates.
    # Within-chunk duplicates were already removed per chunk, so this pass is fast.
    final_df = deduplicate(combined, tolerance_meters=1.5)
    print(f"Final distinct car count: {len(final_df):,}")

    geometry = [Point(xy) for xy in zip(final_df["x"], final_df["y"])]
    gdf      = gpd.GeoDataFrame(final_df, geometry=geometry, crs="EPSG:28992")
    gdf.to_parquet(FINAL_OUTPUT)
    print(f"Saved final result to: {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()