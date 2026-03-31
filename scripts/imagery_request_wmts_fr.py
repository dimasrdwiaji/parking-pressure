# --------------------------------------------
# Parallel aerial imagery WMTS request - France
# IGN Géoplateforme - HR BD Ortho 20cm - Lambert-93
# --------------------------------------------
import os
import math
import asyncio
import aiohttp
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from tqdm.asyncio import tqdm

# --------------------------------------------
# Config
# --------------------------------------------

# Origin read directly from <TopLeftCorner> in the GetCapabilities for
# TileMatrixSet '2154_10cm_10_20'. All zoom levels share the same origin.
ORIGIN_X = 0.0          # EPSG:2154 left edge of the tile grid
ORIGIN_Y = 12000000.0   # EPSG:2154 top edge of the tile grid

# Zoom level 20 is the maximum for this layer.
# Resolution = ScaleDenominator * 0.00028 (OGC standard pixel-size constant)
#            = 714.2857... * 0.00028 = exactly 0.20 m/pixel
ZOOM_LEVEL       = 20
TILE_SIZE_PX     = 256
RESOLUTION       = 714.2857142857143879 * 0.00028  # 0.20 m/pixel
TILE_GROUND_SIZE = RESOLUTION * TILE_SIZE_PX        # 51.2 m per tile side

# --------------------------------------------
# Calculate the right tile to request
# --------------------------------------------
def coords_to_tile(x, y):
    """Converts EPSG:2154 (Lambert-93) coordinates to WMTS tile col and row."""
    col = math.floor((x - ORIGIN_X) / TILE_GROUND_SIZE)
    row = math.floor((ORIGIN_Y - y) / TILE_GROUND_SIZE)
    return col, row

# --------------------------------------------
# Download and save image function
# --------------------------------------------
async def download_tile(session, url, filepath, semaphore):
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()

                    if len(content) == 0:
                        print(f"Empty response: {url}")
                        return

                    with open(filepath, 'wb') as f:
                        f.write(content)
                else:
                    print(f"HTTP {response.status}: {url}")

        except Exception as e:
            print(f"ERROR downloading {url}: {type(e).__name__}: {e}")

# --------------------------------------------
# Main request function
# --------------------------------------------
async def main(parquet_path, out_dir, country_code, max_concurrent_requests=100):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading grid...")

    grid = gpd.read_parquet(parquet_path)
    grid = grid[grid["CNTR_ID"].str.contains(country_code)]

    # Reproject grid to Lambert-93 to match the 2154_10cm_10_20 TileMatrixSet
    grid = grid.to_crs("EPSG:2154")

    grid["minx"] = grid.geometry.bounds.minx
    grid["miny"] = grid.geometry.bounds.miny
    grid["maxx"] = grid.geometry.bounds.maxx
    grid["maxy"] = grid.geometry.bounds.maxy

    print(f"Grid loaded: {len(grid)} cells for {country_code}")

    task_file = f"data/imagery/download_tasks_{country_code}.parquet"

    if os.path.exists(task_file):
        print(f"Loading existing tasks from {task_file}...")
        existing_tasks_df = pd.read_parquet(task_file)
        download_tasks = list(zip(existing_tasks_df['url'], existing_tasks_df['filepath']))

        # Resume fix: skip files already downloaded
        download_tasks = [(url, fp) for url, fp in download_tasks if not os.path.exists(fp)]
        print(f"Remaining tasks after filtering already-downloaded files: {len(download_tasks)}")

    else:
        # Use a set to track unique (tile_col, tile_row) pairs across all grid cells.
        # This prevents downloading the same tile twice when grid cells overlap.
        seen_tiles     = set()
        download_tasks = []

        for idx, grid_row in tqdm(grid.iterrows(), total=len(grid), desc="Preparing URLs"):
            col_min, row_max = coords_to_tile(grid_row.minx, grid_row.miny)
            col_max, row_min = coords_to_tile(grid_row.maxx, grid_row.maxy)

            for tile_col in range(col_min, col_max + 1):
                for tile_row in range(row_min, row_max + 1):

                    # Skip if we already added this tile from a previous grid cell
                    if (tile_col, tile_row) in seen_tiles:
                        continue
                    seen_tiles.add((tile_col, tile_row))

                    # File named by global tile coordinates only — no rel_col/rel_row.
                    # This ensures each unique tile is downloaded exactly once.
                    filename = f"{tile_col}_{tile_row}.jpg"
                    filepath = os.path.join(out_dir, filename)

                    if not os.path.exists(filepath):
                        url = (
                            f"https://data.geopf.fr/wmts"
                            f"?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
                            f"&LAYER=HR.ORTHOIMAGERY.ORTHOPHOTOS.L93"
                            f"&STYLE=normal"
                            f"&FORMAT=image/jpeg"
                            f"&TILEMATRIXSET=2154_10cm_10_20"
                            f"&TILEMATRIX={ZOOM_LEVEL}"
                            f"&TILEROW={tile_row}"
                            f"&TILECOL={tile_col}"
                        )
                        download_tasks.append((url, filepath))

        if download_tasks:
            tasks_df = pd.DataFrame(download_tasks, columns=['url', 'filepath'])
            tasks_df.to_parquet(task_file, index=False)
            print(f"Saved {len(download_tasks)} tasks to {task_file}")
        else:
            print("No tasks to save.")

    print(f"Total tiles to download: {len(download_tasks)}")

    if len(download_tasks) > 0:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        connector = aiohttp.TCPConnector(limit=max_concurrent_requests)
        timeout   = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [download_tile(session, url, fp, semaphore) for url, fp in download_tasks]
            await tqdm.gather(*tasks, desc="Downloading Tiles")
    else:
        print("No new tiles to download.")


if __name__ == "__main__":
    grid_file        = "data/grid_500m_filtered.parquet"
    output_directory = "data/imagery/fr/rgb"
    country_code     = "FR"

    asyncio.run(main(grid_file, output_directory, country_code, max_concurrent_requests=100))