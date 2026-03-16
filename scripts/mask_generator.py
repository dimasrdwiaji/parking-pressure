# Import file
import os
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

parking_gdf = None

def worker_init(parking_path):
    global parking_gdf
    parking_gdf = gpd.read_file(parking_path, layer="combined").to_crs(28992)
    parking_gdf.sindex  # build spatial index


def process_chip(args):

    img_path, masks_dir = args

    chip_name = os.path.basename(img_path).replace(".tif", "")
    mask_path = os.path.join(masks_dir, f"{chip_name}_mask.tif")

    with rasterio.open(img_path) as src:

        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        bounds = src.bounds

        chip_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

        local_parking = parking_gdf[parking_gdf.intersects(chip_polygon)]

        if not local_parking.empty:

            shapes = [(geom, 1) for geom in local_parking.geometry]

            mask = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )

        else:
            mask = np.zeros((height, width), dtype=np.uint8)

    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="lzw"
    ) as dst:

        dst.write(mask, 1)

    return 1


def generate_masks_parallel(max_workers=12):

    images_dir = "data/ams_imagery/images"
    masks_dir = "data/ams_imagery/masks"
    parking_path = "data/parking_space.gpkg"

    os.makedirs(masks_dir, exist_ok=True)

    image_files = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith(".tif")
    ]

    print(f"Found {len(image_files)} chips.")

    tasks = [(img, masks_dir) for img in image_files]

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init,
        initargs=(parking_path,)
    ) as executor:

        list(
            tqdm(
                executor.map(process_chip, tasks),
                total=len(tasks),
                desc="Generating masks"
            )
        )


if __name__ == "__main__":
    generate_masks_parallel(max_workers=12)