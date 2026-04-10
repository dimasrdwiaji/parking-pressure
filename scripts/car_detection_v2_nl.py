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
from collections import defaultdict

# -----------------------------------------------
# Config
# -----------------------------------------------
WMTS_ORIGIN_X    = -285401.92
WMTS_ORIGIN_Y    =  903401.92
RESOLUTION       = 0.21
TILE_SIZE_PX     = 256
TILE_GROUND_SIZE = RESOLUTION * TILE_SIZE_PX

PROMPTS              = ["car", "van", "truck"]
CHUNK_SIZE           = 5000   # number of 2x2 groups per chunk
CONFIDENCE_THRESHOLD = 0.4 # TEST
BATCH_SIZE           = 8
NUM_WORKERS          = 2

IMG_DIR       = "data/imagery/nl/rgb"
OUTPUT_DIR    = "data/imagery/nl/pred"
CHUNKS_DIR    = os.path.join(OUTPUT_DIR, "chunks")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.txt")
FINAL_OUTPUT  = os.path.join(OUTPUT_DIR, "detected_cars_ch1.parquet")


# -----------------------------------------------
# Group tiles into 2x2 blocks
# -----------------------------------------------
def group_tiles(image_paths):
    """
    Groups tile files into 2x2 blocks using tile_col and tile_row from the filename.

    The grouping key is (col // 2, row // 2). Any two tiles that share the same
    key are in the same 2x2 block. For example:
      col=100, row=200 → group key (50, 100)
      col=101, row=200 → group key (50, 100)  ← same group
      col=100, row=201 → group key (50, 100)  ← same group
      col=101, row=201 → group key (50, 100)  ← same group

    We only start groups at even col and even row (col // 2 * 2, row // 2 * 2),
    so every tile belongs to exactly one group with no overlap.

    Groups at country boundaries will have fewer than 4 tiles. Those are kept
    and handled as partial groups — we just stitch whatever tiles are available.
    """
    groups = defaultdict(dict)  # { (group_col, group_row): { (col, row): path } }

    for path in image_paths:
        basename = os.path.basename(path).replace(".jpg", "")
        # Filename format: tile_col_tile_row.jpg (no rel_col/rel_row)
        parts  = basename.rsplit("_", 4)
        col    = int(parts[1])   # actual global tile column
        row    = int(parts[2])   # actual global tile row
        group_key = (col // 2, row // 2)
        groups[group_key][(col, row)] = path

    return groups


def stitch_group(group_tiles_dict, group_key):
    """
    Stitches up to 4 tiles into one 512x512 image.

    The top-left tile of the group is always at:
      col = group_key[0] * 2
      row = group_key[1] * 2

    Each tile is placed at its correct position in the 512x512 canvas:
      top-left:     (col,   row  ) → pixel offset (0,   0  )
      top-right:    (col+1, row  ) → pixel offset (256, 0  )
      bottom-left:  (col,   row+1) → pixel offset (0,   256)
      bottom-right: (col+1, row+1) → pixel offset (256, 256)

    Missing tiles (boundary of coverage area) remain black.
    The function returns the stitched image array and the top-left
    tile coordinates, which are used for geo-coordinate conversion.
    """
    base_col = group_key[0] * 2
    base_row = group_key[1] * 2

    # Start with a black 512x512 canvas
    canvas = np.zeros((TILE_SIZE_PX * 2, TILE_SIZE_PX * 2, 3), dtype=np.uint8)

    for (col, row), path in group_tiles_dict.items():
        # Calculate where this tile sits within the 512x512 canvas
        px_x = (col - base_col) * TILE_SIZE_PX  # 0 or 256
        px_y = (row - base_row) * TILE_SIZE_PX  # 0 or 256

        tile_img = np.array(Image.open(path).convert("RGB"))
        canvas[px_y : px_y + TILE_SIZE_PX, px_x : px_x + TILE_SIZE_PX] = tile_img

    return canvas, base_col, base_row


# -----------------------------------------------
# Dataset — one item per 2x2 group
# -----------------------------------------------
class StitchedTileDataset(Dataset):
    """
    Each item is one 512x512 stitched image made from up to 4 adjacent tiles.
    The dataset receives a list of (group_key, group_tiles_dict) tuples.
    base_col and base_row are the top-left tile coordinates of the group,
    used later to convert pixel positions back to geographic coordinates.
    """
    def __init__(self, groups):
        # groups is a dict: { group_key: { (col, row): path } }
        self.items = list(groups.items())  # list of (group_key, tiles_dict)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        group_key, tiles_dict = self.items[idx]
        img_array, base_col, base_row = stitch_group(tiles_dict, group_key)
        return img_array, base_col, base_row


# -----------------------------------------------
# Per-chunk inference
# -----------------------------------------------
def run_chunk(groups, model, processor, device, text_inputs_per_prompt):
    """
    Processes one chunk of 2x2 stitched groups.

    Each 512x512 image covers the same ground as 4 separate 256x256 tiles,
    but only requires one vision encoder forward pass instead of four.
    This roughly halves the number of encoder calls compared to single tiles,
    since the encoder cost grows sub-linearly with image size.

    Coordinate conversion:
      base_col and base_row are the top-left tile coordinates of the 2x2 group.
      The top-left geographic corner of the stitched image is:
        tile_geo_x = ORIGIN_X + base_col * TILE_GROUND_SIZE
        tile_geo_y = ORIGIN_Y - base_row * TILE_GROUND_SIZE
      Each detected centroid at pixel (cx, cy) within the 512x512 image maps to:
        geo_x = tile_geo_x + cx * RESOLUTION
        geo_y = tile_geo_y - cy * RESOLUTION
      This is identical to the single-tile logic, just with a larger image.
    """
    dataset    = StitchedTileDataset(groups)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    raw_points = []

    with torch.no_grad():
        for images, base_cols, base_rows in tqdm(dataloader, desc="  Groups", leave=False):
            images_list = [img.numpy() for img in images]
            batch_size  = len(images_list)

            # Encode all images in the batch once
            img_inputs = processor(images=images_list, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vision_embeds = model.get_vision_features(
                    pixel_values=img_inputs.pixel_values
                )

            # Run each prompt reusing the cached vision features
            for prompt, text_inputs in text_inputs_per_prompt.items():
                text_inputs_batch = {
                    k: v.expand(batch_size, *v.shape[1:]).to(device)
                    for k, v in text_inputs.items()
                }

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        **text_inputs_batch,
                        vision_embeds=vision_embeds
                    )

                target_sizes = [(img.shape[0], img.shape[1]) for img in images_list]
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=CONFIDENCE_THRESHOLD,
                    target_sizes=target_sizes
                )

                for i in range(batch_size):
                    base_col = base_cols[i].item()
                    base_row = base_rows[i].item()

                    # Top-left geographic corner of the stitched 512x512 image
                    tile_geo_x = WMTS_ORIGIN_X + (base_col * TILE_GROUND_SIZE)
                    tile_geo_y = WMTS_ORIGIN_Y - (base_row * TILE_GROUND_SIZE)

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

    df = pd.DataFrame(raw_points, columns=["x", "y"])
    return deduplicate(df, tolerance_meters=1.5)


# -----------------------------------------------
# Deduplication
# -----------------------------------------------
def deduplicate(df, tolerance_meters=1.5):
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

    # Pre-tokenize prompts once
    print("Pre-tokenizing prompts...")
    text_inputs_per_prompt = {}
    for prompt in PROMPTS:
        inputs = processor(text=prompt, return_tensors="pt")
        text_inputs_per_prompt[prompt] = {
            k: v for k, v in inputs.items() if k != "pixel_values"
        }
    print(f"Prompts ready: {list(text_inputs_per_prompt.keys())}")

    # Load all tiles and apply resume
    all_tiles = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"Total tiles found: {len(all_tiles):,}")

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            done = set(line.strip() for line in f if line.strip())
        remaining = [p for p in all_tiles if p not in done]
        print(f"Resuming — {len(done):,} tiles done, {len(remaining):,} remaining.")
    else:
        done      = set()
        remaining = all_tiles

    if not remaining:
        print("All tiles already processed. Skipping to final merge.")
    else:
        # Group all remaining tiles into 2x2 blocks
        all_groups = group_tiles(remaining)
        group_keys = list(all_groups.keys())
        print(f"Total 2x2 groups to process: {len(group_keys):,}")

        # Split groups into chunks
        chunks = [
            group_keys[i : i + CHUNK_SIZE]
            for i in range(0, len(group_keys), CHUNK_SIZE)
        ]
        chunks = chunks[:1] # TEST FIRST CHUNK
        print(f"Processing {len(chunks)} chunks of up to {CHUNK_SIZE} groups each.")

        for chunk_idx, chunk_keys in enumerate(chunks):
            chunk_num  = len(os.listdir(CHUNKS_DIR))
            chunk_file = os.path.join(CHUNKS_DIR, f"chunk_{chunk_num:05d}.parquet")

            # Build the groups dict for just this chunk
            chunk_groups = {k: all_groups[k] for k in chunk_keys}

            # Collect all tile paths in this chunk for the progress file
            chunk_tile_paths = [
                path
                for tiles_dict in chunk_groups.values()
                for path in tiles_dict.values()
            ]

            print(f"\nChunk {chunk_idx + 1}/{len(chunks)} — "
                  f"{len(chunk_groups)} groups ({len(chunk_tile_paths)} tiles)")

            chunk_df = run_chunk(
                chunk_groups, model, processor, device, text_inputs_per_prompt
            )

            chunk_df.to_parquet(chunk_file, index=False)
            print(f"  Saved {len(chunk_df):,} detections → {chunk_file}")

            # Mark all tiles in this chunk as done
            with open(PROGRESS_FILE, "a") as f:
                for p in chunk_tile_paths:
                    f.write(p + "\n")

    # Final merge and cross-chunk deduplication
    print("\nMerging all chunks...")
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "chunk_*.parquet")))

    if not chunk_files:
        print("No chunk files found. Nothing to merge.")
        return

    combined = pd.concat(
        [pd.read_parquet(cf) for cf in chunk_files], ignore_index=True
    )
    print(f"Total points before final dedup: {len(combined):,}")

    final_df = deduplicate(combined, tolerance_meters=1.2)
    print(f"Final distinct car count: {len(final_df):,}")

    geometry = [Point(xy) for xy in zip(final_df["x"], final_df["y"])]
    gdf      = gpd.GeoDataFrame(final_df, geometry=geometry, crs="EPSG:28992")
    gdf.to_parquet(FINAL_OUTPUT)
    print(f"Saved final result to: {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
