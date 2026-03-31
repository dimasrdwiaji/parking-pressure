import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import requests
from io import BytesIO
from PIL import Image
from shapely.geometry import box

def acquire_imagery_and_masks():
    boundary_path = "data/ams_boundary.gpkg"
    parking_path = "data/parking_space.gpkg"
    out_dir = "data/ams_imagery"
    
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    print("Loading data...")
    ams_boundary = gpd.read_file(boundary_path, layer="Amsterdam").to_crs(28992)
    parking_gdf = gpd.read_file(parking_path).to_crs(28992)
    
    # 1. Define grid size
    minx, miny, maxx, maxy = ams_boundary.total_bounds
    chip_size_px = 512
    gsd = 0.25  # 25cm per pixel
    chip_size_m = chip_size_px * gsd
    
    # Generate original chip grid coordinates
    x_coords = np.arange(minx, maxx, chip_size_m)
    y_coords = np.arange(miny, maxy, chip_size_m)
    
    # Number of chips in each direction
    n_chips_x = len(x_coords) - 1   # because we use x_coords[:-1] later
    n_chips_y = len(y_coords) - 1
    
    # Block size: 4 chips per side (2048 pixels)
    chips_per_block = 4
    block_size_m = chips_per_block * chip_size_m
    
    # Number of blocks (rounding up to cover all chips)
    n_blocks_x = (n_chips_x + chips_per_block - 1) // chips_per_block
    n_blocks_y = (n_chips_y + chips_per_block - 1) // chips_per_block
    
    wms_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
    
    print(f"Original chip grid: {n_chips_x} x {n_chips_y} = {n_chips_x * n_chips_y} chips")
    print(f"Block grid: {n_blocks_x} x {n_blocks_y} = {n_blocks_x * n_blocks_y} blocks")
    print("Beginning download and rasterization process...")
    
    total_chips_processed = 0
    
    for block_ix in range(n_blocks_x):
        for block_iy in range(n_blocks_y):
            # Chip index range for this block
            i_start = block_ix * chips_per_block
            i_end = min(i_start + chips_per_block, n_chips_x)
            j_start = block_iy * chips_per_block
            j_end = min(j_start + chips_per_block, n_chips_y)
            
            # Block bounding box (covers all chips in this block)
            block_left = x_coords[i_start]
            block_right = x_coords[i_end] + chip_size_m   # right edge of last chip
            block_bottom = y_coords[j_start]
            block_top = y_coords[j_end] + chip_size_m     # top edge of last chip
            block_bbox = (block_left, block_bottom, block_right, block_top)
            
            block_width_px = (i_end - i_start) * chip_size_px
            block_height_px = (j_end - j_start) * chip_size_px
            
            # Resume: check if all chips in this block already exist
            all_chips_exist = True
            for di in range(i_end - i_start):
                for dj in range(j_end - j_start):
                    chip_i = i_start + di
                    chip_j = j_start + dj
                    chip_name = f"chip_{chip_i}_{chip_j}"
                    img_path = os.path.join(out_dir, "images", f"{chip_name}.tif")
                    mask_path = os.path.join(out_dir, "masks", f"{chip_name}_mask.tif")
                    if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                        all_chips_exist = False
                        break
                if not all_chips_exist:
                    break
            
            if all_chips_exist:
                print(f"Block {block_ix},{block_iy} already complete, skipping.")
                total_chips_processed += (i_end - i_start) * (j_end - j_start)
                continue
            
            print(f"Processing block {block_ix},{block_iy} (chips {i_start}:{i_end-1}, {j_start}:{j_end-1})")
            
            # WMS request for the block
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
                response = requests.get(wms_url, params=params, timeout=60)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"  Network error for block {block_ix},{block_iy}: {e}. Skipping block.")
                continue
            
            if response.status_code != 200:
                print(f"  Bad status code {response.status_code} for block {block_ix},{block_iy}")
                continue
            
            # Convert block JPEG to numpy array
            img = Image.open(BytesIO(response.content))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            block_img = np.array(img)  # shape: (height, width, 3)
            
            # Geotransform for the block (top-left origin)
            block_transform = from_origin(west=block_left, north=block_top, xsize=gsd, ysize=gsd)
            
            # Rasterize parking mask for the entire block
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
            
            # Split block into chips
            for di in range(i_end - i_start):
                for dj in range(j_end - j_start):
                    chip_i = i_start + di
                    chip_j = j_start + dj
                    chip_name = f"chip_{chip_i}_{chip_j}"
                    img_path = os.path.join(out_dir, "images", f"{chip_name}.tif")
                    mask_path = os.path.join(out_dir, "masks", f"{chip_name}_mask.tif")
                    
                    # Pixel coordinates of this chip within the block
                    px = di * chip_size_px
                    py = dj * chip_size_px
                    
                    # Extract chip image and mask
                    chip_img = block_img[py:py+chip_size_px, px:px+chip_size_px, :]
                    chip_mask = block_mask[py:py+chip_size_px, px:px+chip_size_px]
                    
                    # Compute chip geotransform (from its own bounds)
                    chip_left = block_left + di * chip_size_m
                    chip_top = block_top - dj * chip_size_m  # because y decreases downward
                    chip_transform = from_origin(west=chip_left, north=chip_top, xsize=gsd, ysize=gsd)
                    
                    # Save chip image
                    with rasterio.open(
                        img_path, 'w', driver='GTiff',
                        height=chip_size_px, width=chip_size_px,
                        count=3, dtype=chip_img.dtype,
                        crs='EPSG:28992', transform=chip_transform,
                    ) as dst:
                        for band in range(3):
                            dst.write(chip_img[:, :, band], band + 1)
                    
                    # Save chip mask
                    with rasterio.open(
                        mask_path, 'w', driver='GTiff',
                        height=chip_size_px, width=chip_size_px,
                        count=1, dtype=chip_mask.dtype,
                        crs='EPSG:28992', transform=chip_transform,
                    ) as dst:
                        dst.write(chip_mask, 1)
                    
                    total_chips_processed += 1
                    
                    if total_chips_processed % 100 == 0:
                        print(f"  ... {total_chips_processed} chips saved so far")
    
    print(f"Image acquisition and mask rasterization complete. Total chips processed: {total_chips_processed}")

if __name__ == "__main__":
    acquire_imagery_and_masks()