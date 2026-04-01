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
# Clear CUDA memory
# -----------------------------------------------
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# -----------------------------------------------
# Config
# -----------------------------------------------
WMTS_ORIGIN_X    = -285401.92
WMTS_ORIGIN_Y    =  903401.92
RESOLUTION       = 0.21
TILE_SIZE_PX     = 256
TILE_GROUND_SIZE = RESOLUTION * TILE_SIZE_PX

PROMPTS              = ["car", "van", "truck"]
CHUNK_SIZE           = 5000
CONFIDENCE_THRESHOLD = 0.5

IMG_DIR       = "data/imagery/nl/rgb"
OUTPUT_DIR    = "data/imagery/nl/pred"
CHUNKS_DIR    = os.path.join(OUTPUT_DIR, "chunks")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.txt")
FINAL_OUTPUT  = os.path.join(OUTPUT_DIR, "detected_cars_final.parquet")


# -----------------------------------------------
# Dataset
# -----------------------------------------------
class WMTSTileDataset(Dataset):
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
def run_chunk(image_paths, model, processor, device, text_inputs_per_prompt):
    """
    Processes one chunk of tiles.

    What is pre-computed before this function (in main):
      - text_inputs_per_prompt: a dict of { prompt: tokenized_text_inputs }
        The processor tokenizes each prompt once. Tokenization is CPU-only and
        cheap, but doing it millions of times across batches adds up. We do it
        once per prompt and reuse the token tensors every batch.

    Inside the batch loop:
      - Vision features are computed ONCE per batch with get_vision_features().
        This is the expensive part (~450M param encoder). It runs once regardless
        of how many prompts we use.
      - For each prompt, we call model() with the cached vision features plus
        the pre-tokenized text inputs. The model() call here only needs to run
        the text encoder + decoder, skipping re-encoding the image.

    Why not pre-compute text *embeddings* (get_text_features)?
      The SAM3 forward() does not accept raw text embeddings as a kwarg in a
      way that's compatible with how the DETR cross-attention expects them.
      Passing vision_embeds and text_embeds separately causes a shape mismatch
      inside the cross-attention layer (confirmed by the 1024x256 error).
      The safe approach is to let model() handle the text encoding internally,
      but feed it pre-tokenized inputs so at least tokenization is not repeated.
    """
    dataset    = WMTSTileDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    raw_points = []

    with torch.no_grad():
        for images, cols, rows, paths in tqdm(dataloader, desc="  Tiles", leave=False):
            images_list = [img.numpy() for img in images]
            batch_size  = len(images_list)

            # --- Encode images ONCE per batch ---
            img_inputs = processor(images=images_list, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vision_embeds = model.get_vision_features(
                    pixel_values=img_inputs.pixel_values
                )

            # --- Run each prompt using cached vision features ---
            for prompt, text_inputs in text_inputs_per_prompt.items():

                # Text was tokenized once. We just need to expand the token
                # tensors to match the current batch size.
                # Some batches (last batch of a chunk) may be smaller than 32.
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

    # --- Pre-tokenize text prompts ONCE ---
    # Tokenization is CPU-only and deterministic. Running it once per prompt
    # and reusing the token tensors avoids redundant work across all batches.
    # We store with batch size 1 and expand to actual batch size inside run_chunk.
    print("Pre-tokenizing prompts...")
    text_inputs_per_prompt = {}
    for prompt in PROMPTS:
        inputs = processor(text=prompt, return_tensors="pt")
        # Store on CPU — we move to device inside run_chunk per batch
        text_inputs_per_prompt[prompt] = {k: v for k, v in inputs.items()
                                          if k not in ("pixel_values",)}
    print(f"Prompts ready: {list(text_inputs_per_prompt.keys())}")

    # --- Load tiles and apply resume ---
    all_tiles = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"Total tiles found: {len(all_tiles):,}")

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            done = set(line.strip() for line in f if line.strip())
        remaining = [p for p in all_tiles if p not in done]
        print(f"Resuming — {len(done):,} done, {len(remaining):,} remaining.")
    else:
        done      = set()
        remaining = all_tiles

    if not remaining:
        print("All tiles already processed. Skipping to final merge.")
    else:
        chunks = [
            remaining[i : i + CHUNK_SIZE]
            for i in range(0, len(remaining), CHUNK_SIZE)
        ]
        print(f"Processing {len(chunks)} chunks of up to {CHUNK_SIZE} tiles each.")

        for chunk_idx, chunk_paths in enumerate(chunks):
            # Use existing file count so we never overwrite previous chunk files
            chunk_num  = len(os.listdir(CHUNKS_DIR))
            chunk_file = os.path.join(CHUNKS_DIR, f"chunk_{chunk_num:05d}.parquet")

            print(f"\nChunk {chunk_idx + 1}/{len(chunks)} — {len(chunk_paths)} tiles")

            chunk_df = run_chunk(
                chunk_paths, model, processor, device, text_inputs_per_prompt
            )

            chunk_df.to_parquet(chunk_file, index=False)
            print(f"  Saved {len(chunk_df):,} detections → {chunk_file}")

            with open(PROGRESS_FILE, "a") as f:
                for p in chunk_paths:
                    f.write(p + "\n")

    # --- Final merge and cross-chunk deduplication ---
    print("\nMerging all chunks...")
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "chunk_*.parquet")))

    if not chunk_files:
        print("No chunk files found. Nothing to merge.")
        return

    combined = pd.concat(
        [pd.read_parquet(cf) for cf in chunk_files], ignore_index=True
    )
    print(f"Total points before final dedup: {len(combined):,}")

    final_df = deduplicate(combined, tolerance_meters=1.5)
    print(f"Final distinct car count: {len(final_df):,}")

    geometry = [Point(xy) for xy in zip(final_df["x"], final_df["y"])]
    gdf      = gpd.GeoDataFrame(final_df, geometry=geometry, crs="EPSG:28992")
    gdf.to_parquet(FINAL_OUTPUT)
    print(f"Saved final result to: {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()