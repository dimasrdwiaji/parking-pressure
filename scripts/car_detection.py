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
TILE_GROUND_SIZE = RESOLUTION * TILE_SIZE_PX  # 53.76 meters

class WMTSTileDataset(Dataset):
    """Custom Dataset to load JPEGs and parse their geographic matrix indices from the filename."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Filename format: GridID_Col_Row.jpg
        basename = os.path.basename(path).replace(".jpg", "")
        parts = basename.rsplit("_", 4) # safety reason, in case GRD_ID value already has underscore
        
        col = int(parts[1])
        row = int(parts[2])
        
        # Load image
        img = Image.open(path).convert("RGB")
        img_array = np.array(img)
        
        return img_array, col, row

def extract_centroids_from_masks(boolean_masks):
    """Calculates the [Y, X] pixel centroid for each detected instance mask."""
    centroids = []
    # boolean_masks shape: (num_instances, H, W)
    for mask in boolean_masks:
        if mask.sum() > 0:  # If the mask isn't empty
            # center_of_mass returns (Y, X)
            cy, cx = center_of_mass(mask)
            centroids.append((cx, cy))
    return centroids

def run_inference_and_extract(img_dir, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM 3 to {device}...")
    
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
    model.eval()

    image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"Found {len(image_files):,} tiles for processing.")
    
    dataset = WMTSTileDataset(image_files)
    # num_workers=4 keeps the CPU loading JPEGs ahead of the GPU
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    master_point_list = []

    with torch.no_grad():
        for images, cols, rows in tqdm(dataloader, desc="Detecting Cars"):
            # 'images' is a batch of numpy arrays. We need them as a list for the processor.
            images_list = [img.numpy() for img in images]
            text_prompts = ["car"] * len(images_list)
            
            # Prepare inputs
            inputs = processor(
                images=images_list,
                text=text_prompts,
                return_tensors="pt"
            ).to(device)
            
            # Run Mixed Precision Inference
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
            
            # Extract masks
            target_sizes = [(img.shape[0], img.shape[1]) for img in images_list]
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,        # Confidence score — filters ghost/background queries
                target_sizes=target_sizes
            )
            
            # Process each image in the batch
            for i in range(len(images_list)):
                col = cols[i].item()
                row = rows[i].item()
                
                # Calculate the top-left geographic coordinate of this specific WMTS tile
                tile_geo_x = ORIGIN_X + (col * TILE_GROUND_SIZE)
                tile_geo_y = ORIGIN_Y - (row * TILE_GROUND_SIZE)
                
                # Get the binary masks for this specific image
                masks_np = results[i]["masks"].cpu().numpy()
                
                for mask in masks_np:
                    if mask.sum() == 0:
                        continue
                    
                    cy, cx = center_of_mass(mask.astype(np.uint8))

                    geo_x = tile_geo_x + (cx * RESOLUTION)
                    geo_y = tile_geo_y - (cy * RESOLUTION)
                    master_point_list.append([geo_x, geo_y])
                
                ## Get the binary masks for this specific image
                #img_masks = batch_masks[i]
                
                ## Get [X, Y] pixel coordinates of the cars
                #car_pixels = extract_centroids_from_masks(img_masks)
                
                # Convert Pixels -> Real-world Coordinates
                #for px_x, px_y in car_pixels:
                #    geo_x = tile_geo_x + (px_x * RESOLUTION)
                #    geo_y = tile_geo_y - (px_y * RESOLUTION)
                   
                #    master_point_list.append([geo_x, geo_y])

    print(f"Inference complete. Extracted {len(master_point_list):,} raw car centroids.")
    return pd.DataFrame(master_point_list, columns=['x', 'y'])

def deduplicate_cars(df, tolerance_meters=1.5):
    """Removes double-counted cars sitting on tile boundaries using spatial clustering."""
    print("Deduplicating overlapping detections...")
    
    coords = df[['x', 'y']].values
    
    # DBSCAN clusters points that are within 'eps' distance of each other.
    # min_samples=1 means a car sitting by itself forms its own cluster of 1.
    clustering = DBSCAN(eps=tolerance_meters, min_samples=1, n_jobs=-1).fit(coords)
    
    df['cluster_id'] = clustering.labels_
    
    # Take the mean geographic coordinate of each cluster
    df_clean = df.groupby('cluster_id').mean().reset_index(drop=True)
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_clean['x'], df_clean['y'])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:28992")
    
    reduction = (1 - (len(gdf) / len(df))) * 100
    print(f"Deduplication complete! Removed {reduction:.2f}% of points (boundary overlaps).")
    print(f"Final distinct car count: {len(gdf):,}")
    
    return gdf

if __name__ == "__main__":
    tile_directory = "data/imagery/nl/rgb"
    output_geoparquet = "data/imagery/nl/pred/detected_cars.parquet"
    
    # 1. Run GPU Inference
    raw_cars_df = run_inference_and_extract(tile_directory, batch_size=4)
    
    # 2. Deduplicate
    if not raw_cars_df.empty:
        final_gdf = deduplicate_cars(raw_cars_df, tolerance_meters=1.5)
        
        # 3. Save as GeoParquet for the next phase
        final_gdf.to_parquet(output_geoparquet)
        print(f"Saved highly-optimized point cloud to {output_geoparquet}")
    else:
        print("No cars detected in the dataset.")