import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
import requests
from io import BytesIO
from PIL import Image


def acquire_imagery_and_masks():
    boundary_path = "data/ams_boundary.gpkg"
    parking_path = "data/parking_space.gpkg"
    out_dir = "data/ams_imagery"
    
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    print("Loading data...")
    ams_boundary = gpd.read_file(boundary_path, layer = "Amsterdam").to_crs(28992)
    parking_gdf = gpd.read_file(parking_path).to_crs(28992)
    
    # 1. Define grid size
    minx, miny, maxx, maxy = ams_boundary.total_bounds
    chip_size_px = 512
    gsd = 0.25 # 25cm per pixel
    chip_size_m = chip_size_px * gsd
    
    # Generate grid coordinates
    x_coords = np.arange(minx, maxx, chip_size_m)
    y_coords = np.arange(miny, maxy, chip_size_m)
    
    wms_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
    
    print(f"Generated grid: {len(x_coords) * len(y_coords)} potential chips.")
    print("Beginning download and rasterization process...")
    chip_count = 0
    
    # Request imagery per grid
    for i, x in enumerate(x_coords[:-1]):
        for j, y in enumerate(y_coords[:-1]):
                
            chip_bbox = (x, y, x + chip_size_m, y + chip_size_m)
            chip_name = f"chip_{i}_{j}"
            img_path = os.path.join(out_dir, "images", f"{chip_name}.tif")
            mask_path = os.path.join(out_dir, "masks", f"{chip_name}_mask.tif")
            
            # Resume mechanism
            if os.path.exists(img_path) and os.path.exists(mask_path):
                chip_count += 1
                continue
            
            print(f"Processing {chip_name}")
            # WMS Request for Image
            # Define parameter
            params = {
                "SERVICE": "WMS",
                "VERSION": "1.3.0",
                "REQUEST": "GetMap",
                "FORMAT": "image/jpeg", # does not allow tif
                "TRANSPARENT": "true",
                "LAYERS": "2025_ortho25", # latest 25cm layer
                "CRS": "EPSG:28992",
                "STYLES": "",
                "WIDTH": str(chip_size_px),
                "HEIGHT": str(chip_size_px),
                "BBOX": f"{chip_bbox[0]},{chip_bbox[1]},{chip_bbox[2]},{chip_bbox[3]}"
            }
            
            response = requests.get(wms_url, params=params)
            if response.status_code != 200:
                continue
            
            # Convert to tiff
            # Read the JPEG image data from the response
            img = Image.open(BytesIO(response.content))
            
            # Convert to numpy array (shape: height, width, bands)
            img_array = np.array(img)
            
            # Configure transformation parameters
            transform = from_origin(west=chip_bbox[0], north=chip_bbox[3], xsize=gsd, ysize=gsd)
                
            img_path = os.path.join(out_dir, "images", f"{chip_name}.tif")
            with rasterio.open(
                img_path, 'w', driver='GTiff',
                height=img_array.shape[0],
                width=img_array.shape[1],
                count=img_array.shape[2],  # Number of bands (e.g., 3 for RGB)
                dtype=img_array.dtype,
                crs='EPSG:28992',
                transform=transform,
            ) as dst:
                for band in range(img_array.shape[2]):
                    dst.write(img_array[:, :, band], band + 1)
                    
            # 3. Create matched transform and rasterize mask
            # The WMS returns data top-down, so max Y is the top edge
            transform = from_bounds(chip_bbox[0], chip_bbox[1], chip_bbox[2], chip_bbox[3], chip_size_px, chip_size_px)
            
            # Filter polygons that intersect this specific chip
            from shapely.geometry import box
            chip_polygon = box(*chip_bbox)
            local_parking = parking_gdf[parking_gdf.intersects(chip_polygon)]
            
            # Rasterize
            mask_shape = (chip_size_px, chip_size_px)
            if not local_parking.empty:
                # Pair geometries with a value of 1 for parking
                geom_value_pairs = [(geom, 1) for geom in local_parking.geometry]
                mask = rasterize(
                    shapes=geom_value_pairs,
                    out_shape=mask_shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
            else:
                mask = np.zeros(mask_shape, dtype=np.uint8)
                
            # Save mask
            mask_path = os.path.join(out_dir, "masks", f"{chip_name}_mask.tif")
            with rasterio.open(
                mask_path, 'w', driver='GTiff',
                height=mask.shape[0], width=mask.shape[1],
                count=1, dtype=mask.dtype,
                crs='EPSG:28992', transform=transform,
            ) as dst:
                dst.write(mask, 1)
                
            chip_count += 1

    print("Image acquisition and mask rasterization complete.")

if __name__ == "__main__":
    acquire_imagery_and_masks()