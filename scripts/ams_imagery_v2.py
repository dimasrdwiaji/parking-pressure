import os
import time
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
from PIL import Image
from shapely.geometry import box
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def create_retry_session():
    """Creates a requests session with automatic retries for robust WMS downloading."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def process_block(block_ix, block_iy, x_coords, y_coords, n_chips_x, n_chips_y, 
                  chip_size_px, chip_size_m, gsd, chips_per_block, wms_url, 
                  out_dir, parking_gdf):
    """Worker function to process a single 4x4 block."""
    session = create_retry_session()
    
    # Calculate chip index boundaries for this block
    i_start = block_ix * chips_per_block
    i_end = min(i_start + chips_per_block, n_chips_x)
    j_start = block_iy * chips_per_block
    j_end = min(j_start + chips_per_block, n_chips_y)
    
    n_chips_in_x = i_end - i_start
    n_chips_in_y = j_end - j_start
    
    # 1. Bulletproof Block Bounding Box (fixes the off-by-one spatial squash)
    block_left = x_coords[i_start]
    block_right = block_left + (n_chips_in_x * chip_size_m)
    block_bottom = y_coords[j_start]
    block_top = block_bottom + (n_chips_in_y * chip_size_m)
    block_bbox = (block_left, block_bottom, block_right, block_top)
    
    block_width_px = n_chips_in_x * chip_size_px
    block_height_px = n_chips_in_y * chip_size_px
    
    # 2. Resume Check
    all_chips_exist = True
    for di in range(n_chips_in_x):
        for dj_img in range(n_chips_in_y):
            chip_i = i_start + di
            # Fix vertical inversion: map top image row (0) to highest geographic y-index
            chip_j = (j_end - 1) - dj_img
            
            img_path = os.path.join(out_dir, "images", f"chip_{chip_i}_{chip_j}.tif")
            mask_path = os.path.join(out_dir, "masks", f"chip_{chip_i}_{chip_j}_mask.tif")
            
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                all_chips_exist = False
                break
        if not all_chips_exist:
            break
            
    if all_chips_exist:
        return n_chips_in_x * n_chips_in_y

    # 3. WMS Request
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "FORMAT": "image/jpeg",
        "TRANSPARENT": "true",
        "LAYERS": "2025_ortho25",
        "CRS": "EPSG:28992",
        "STYLES": "",
        "WIDTH": str(block_width_px),
        "HEIGHT": str(block_height_px),
        "BBOX": f"{block_left},{block_bottom},{block_right},{block_top}"
    }
    
    try:
        response = session.get(wms_url, params=params, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"\n[Error] WMS failed for block {block_ix},{block_iy}: {e}")
        return 0
        
    # Process JPEG
    img = Image.open(BytesIO(response.content))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    block_img = np.array(img)
    
    # 4. Rasterize Block Mask
    block_transform = from_origin(west=block_left, north=block_top, xsize=gsd, ysize=gsd)
    block_polygon = box(*block_bbox)
    local_parking = parking_gdf[parking_gdf.intersects(block_polygon)]
    
    if not local_parking.empty:
        geom_value_pairs = [(geom, 1) for geom in local_parking.geometry]
        block_mask = rasterize(
            shapes=geom_value_pairs,
            out_shape=(block_height_px, block_width_px),
            transform=block_transform,
            fill=0,
            dtype=np.uint8
        )
    else:
        block_mask = np.zeros((block_height_px, block_width_px), dtype=np.uint8)
        
    # 5. Split and Save
    chips_saved = 0
    for di in range(n_chips_in_x):
        for dj_img in range(n_chips_in_y): # dj_img goes 0 (top) to max (bottom)
            chip_i = i_start + di
            chip_j = (j_end - 1) - dj_img # Correct geographical pairing
            
            px = di * chip_size_px
            py = dj_img * chip_size_px
            
            chip_img = block_img[py:py+chip_size_px, px:px+chip_size_px, :]
            chip_mask = block_mask[py:py+chip_size_px, px:px+chip_size_px]
            
            # Exact transform for this specific chip
            chip_left = block_left + (di * chip_size_m)
            chip_top = block_top - (dj_img * chip_size_m)
            chip_transform = from_origin(west=chip_left, north=chip_top, xsize=gsd, ysize=gsd)
            
            img_path = os.path.join(out_dir, "images", f"chip_{chip_i}_{chip_j}.tif")
            mask_path = os.path.join(out_dir, "masks", f"chip_{chip_i}_{chip_j}_mask.tif")
            
            with rasterio.open(
                img_path, 'w', driver='GTiff', height=chip_size_px, width=chip_size_px,
                count=3, dtype=chip_img.dtype, crs='EPSG:28992', transform=chip_transform, compress='lzw'
            ) as dst:
                for band in range(3):
                    dst.write(chip_img[:, :, band], band + 1)
                    
            with rasterio.open(
                mask_path, 'w', driver='GTiff', height=chip_size_px, width=chip_size_px,
                count=1, dtype=chip_mask.dtype, crs='EPSG:28992', transform=chip_transform, compress='lzw'
            ) as dst:
                dst.write(chip_mask, 1)
                
            chips_saved += 1
            
    return chips_saved

def acquire_imagery_and_masks_parallel(max_workers=8):
    boundary_path = "data/ams_boundary.gpkg"
    parking_path = "data/parking_space.gpkg"
    out_dir = "data/ams_imagery"
    
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    print("Loading geographic vector data...")
    ams_boundary = gpd.read_file(boundary_path).to_crs(28992)
    parking_gdf = gpd.read_file(parking_path).to_crs(28992)
    
    minx, miny, maxx, maxy = ams_boundary.total_bounds
    chip_size_px = 512
    gsd = 0.25 
    chip_size_m = chip_size_px * gsd
    
    x_coords = np.arange(minx, maxx, chip_size_m)
    y_coords = np.arange(miny, maxy, chip_size_m)
    
    n_chips_x = len(x_coords) - 1
    n_chips_y = len(y_coords) - 1
    
    chips_per_block = 4
    n_blocks_x = (n_chips_x + chips_per_block - 1) // chips_per_block
    n_blocks_y = (n_chips_y + chips_per_block - 1) // chips_per_block
    
    wms_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
    
    print(f"Total Chips: {n_chips_x * n_chips_y}")
    print(f"Total Blocks (4x4 chips): {n_blocks_x * n_blocks_y}")
    print(f"Starting parallel download with {max_workers} threads...")

    # Build the task list
    tasks = []
    for block_ix in range(n_blocks_x):
        for block_iy in range(n_blocks_y):
            tasks.append((
                block_ix, block_iy, x_coords, y_coords, n_chips_x, n_chips_y, 
                chip_size_px, chip_size_m, gsd, chips_per_block, wms_url, out_dir, parking_gdf
            ))

    total_chips_processed = 0
    start_time = time.time()

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_block, *task): task for task in tasks}
        
        # tqdm progress bar tracks completed blocks
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Blocks"):
            try:
                result = future.result()
                total_chips_processed += result
            except Exception as e:
                print(f"A block processing thread crashed: {e}")

    elapsed = time.time() - start_time
    print(f"Pipeline complete! Saved {total_chips_processed} chips in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    # Adjust max_workers depending on network strength (start with 12, increase to around 16 if the WMS server allows)
    acquire_imagery_and_masks_parallel(max_workers=12)