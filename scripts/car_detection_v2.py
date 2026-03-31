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

# --- WMTS Constants (Level 14) ---
ORIGIN_X = -285401.92
ORIGIN_Y = 903401.92
RESOLUTION = 0.21
TILE_SIZE_PX = 256
TILE_GROUND_SIZE = RESOLUTION * TILE_SIZE_PX  # 53.76 metres

VEHICLE_PROMPTS = ["car", "truck", "van", "bus"]


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
        img = Image.open(path).convert("RGB")
        return np.array(img), col, row


def build_text_embed_cache(model, processor, prompts, device):
    cache = {}
    for prompt in prompts:
        text_inputs = processor(text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            text_output = model.get_text_features(**text_inputs)

        # CLIPTextModelOutput — extract the full hidden state sequence,
        # NOT pooler_output: the mask decoder cross-attends to all tokens
        text_embeds = text_output.last_hidden_state  # (1, seq_len, hidden_dim)

        cache[prompt] = {
            "text_embeds": text_embeds,
            "attention_mask": text_inputs.attention_mask   # (1, seq_len)
        }
        print(f"  Cached text embed for '{prompt}': {text_embeds.shape}")
    return cache

def run_inference_and_extract(img_dir, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 to {device}...")

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained(
        "facebook/sam3", torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # --- Pre-compute text embeddings once for all prompts ---
    print("Pre-computing text embeddings...")
    text_cache = build_text_embed_cache(model, processor, VEHICLE_PROMPTS, device)

    image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"Found {len(image_files):,} tiles | Prompts: {VEHICLE_PROMPTS}")

    dataset = WMTSTileDataset(image_files)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    master_point_list = []

    with torch.no_grad():
        for images, cols, rows in tqdm(dataloader, desc="Detecting Vehicles"):
            images_list = [img.numpy() for img in images]
            batch_size_actual = len(images_list)

            # --- Vision encoding: fresh per tile batch (text-conditioned internally) ---
            img_inputs = processor(images=images_list, return_tensors="pt").to(device)
            target_sizes = img_inputs.get("original_sizes").tolist()

            # Per-image mask accumulator, reset each batch
            accumulated_masks = [[] for _ in range(batch_size_actual)]

            # --- Text prompts: re-use cached embeddings, expand to batch size ---
            for prompt in VEHICLE_PROMPTS:
                cached = text_cache[prompt]

                # Expand (1, seq_len, D) → (batch_size, seq_len, D)
                text_embeds = cached["text_embeds"].expand(batch_size_actual, -1, -1)
                attention_mask = cached["attention_mask"].expand(batch_size_actual, -1)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        pixel_values=img_inputs.pixel_values,
                        text_embeds=text_embeds,
                        attention_mask=attention_mask,
                    )

                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=target_sizes,
                )

                for i, result in enumerate(results):
                    for mask in result["masks"]:
                        accumulated_masks[i].append(mask.cpu().numpy())

            # --- Centroid extraction from all prompts' masks ---
            for i in range(batch_size_actual):
                col = cols[i].item()
                row = rows[i].item()

                tile_geo_x = ORIGIN_X + (col * TILE_GROUND_SIZE)
                tile_geo_y = ORIGIN_Y - (row * TILE_GROUND_SIZE)

                for mask_np in accumulated_masks[i]:
                    if mask_np.sum() == 0:
                        continue
                    cy, cx = center_of_mass(mask_np.astype(np.uint8))
                    geo_x = tile_geo_x + (cx * RESOLUTION)
                    geo_y = tile_geo_y - (cy * RESOLUTION)
                    master_point_list.append([geo_x, geo_y])

    print(f"Inference complete. Extracted {len(master_point_list):,} raw centroids.")
    return pd.DataFrame(master_point_list, columns=["x", "y"])


def deduplicate_cars(df, tolerance_meters=1.5):
    """
    Collapses both:
    - tile-boundary duplicates (same vehicle in two adjacent tiles)
    - cross-prompt duplicates (same vehicle matched by 'car' AND 'van')
    """
    print("Deduplicating overlapping detections...")
    coords = df[["x", "y"]].values
    clustering = DBSCAN(eps=tolerance_meters, min_samples=1, n_jobs=-1).fit(coords)
    df["cluster_id"] = clustering.labels_

    df_clean = df.groupby("cluster_id").mean().reset_index(drop=True)
    geometry = [Point(xy) for xy in zip(df_clean["x"], df_clean["y"])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:28992")

    reduction = (1 - (len(gdf) / len(df))) * 100
    print(f"Removed {reduction:.2f}% duplicate points.")
    print(f"Final distinct vehicle count: {len(gdf):,}")
    return gdf


if __name__ == "__main__":
    tile_directory = "data/imagery/nl/rgb"
    output_geoparquet = "data/imagery/nl/pred/detected_cars_v2.parquet"

    raw_df = run_inference_and_extract(tile_directory, batch_size=4)

    if not raw_df.empty:
        final_gdf = deduplicate_cars(raw_df, tolerance_meters=1.5)
        final_gdf.to_parquet(output_geoparquet)
        print(f"Saved to {output_geoparquet}")
    else:
        print("No vehicles detected.")
