# --------------------------------------------
# Parallel aerial imagery WMTS request.
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
ORIGIN_X = -285401.92 # RD New format
ORIGIN_Y = 903401.92 # RD New format
RESOLUTION = 0.21 # Tile 14
TILE_SIZE_PX = 256
TILE_GROUND_SIZE = RESOLUTION * TILE_SIZE_PX

# --------------------------------------------
# Calculate the right tile to request
# --------------------------------------------
def coords_to_tile(x, y):
    """Converts RD new coordinates to WMTS column and row."""
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
    
    # Load grid parquet
    grid = gpd.read_parquet(parquet_path)
    
    print("Grid loaded.")
    
    # FIX 1: You MUST assign the filtered result back to 'grid'
    grid = grid[grid["CNTR_ID"].str.contains(country_code)]
    
    # Reproject grid
    grid = grid.to_crs("EPSG:28992")
    
    # Extract grid boundary
    grid["minx"] = grid.geometry.bounds.minx
    grid["miny"] = grid.geometry.bounds.miny
    grid["maxx"] = grid.geometry.bounds.maxx
    grid["maxy"] = grid.geometry.bounds.maxy
    
    # Check if download_task file exist
    task_file = f"data/imagery/download_tasks_{country_code}.parquet"
    
        # Load existing tasks or create new list
    if os.path.exists(task_file):
        print(f"Loading existing tasks from {task_file}...")
        existing_tasks_df = pd.read_parquet(task_file)
        # Convert to list of tuples if needed for download
        download_tasks = list(zip(existing_tasks_df['url'], existing_tasks_df['filepath']))
        
        
        print(f"Loaded {len(download_tasks)} existing tasks")
    else:
        # Create new task list
        download_tasks = []
        
        # Iterate over grid
        for idx, grid_row in tqdm(grid.iterrows(), total=len(grid), desc="Preparing URLs"):
            col_min, row_max = coords_to_tile(grid_row.minx, grid_row.miny)
            col_max, row_min = coords_to_tile(grid_row.maxx, grid_row.maxy)
            grid_id = grid_row.get('GRD_ID', f"Grid_{idx}")

            for tile_col in range(col_min, col_max + 1):
                for tile_row in range(row_min, row_max + 1):
                    rel_col = tile_col - col_min
                    rel_row = tile_row - row_min

                    filename = f"{grid_id}_{tile_col}_{tile_row}_{rel_col}_{rel_row}.jpg"
                    filepath = os.path.join(out_dir, filename)

                    # Skip if file already exists
                    if not os.path.exists(filepath):
                        url = (
                            f"https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0"
                            f"?Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg"
                            f"&Layer=2025_ortho25&Style=default&TileMatrixSet=EPSG:28992"
                            f"&TileMatrix=14&TileCol={tile_col}&TileRow={tile_row}"
                        )
                        download_tasks.append((url, filepath))
        
        # Save the tasks to parquet for future use
        if download_tasks:
            tasks_df = pd.DataFrame(download_tasks, columns=['url', 'filepath'])
            tasks_df.to_parquet(task_file, index=False)
            print(f"Saved {len(download_tasks)} tasks to {task_file}")
        else:
            print("No tasks to save")

    print(f"Total tiles to request: {len(download_tasks)}")
    # download_tasks = download_tasks[:1000] # Get the first thousand images to check with later workflow
    
    # Run the request
    if len(download_tasks) > 0:
        semaphore = asyncio.Semaphore(max_concurrent_requests) 
        connector = aiohttp.TCPConnector(limit=max_concurrent_requests) 
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [download_tile(session, url, fp, semaphore) for url, fp in download_tasks]
            await tqdm.gather(*tasks, desc="Downloading Tiles")
    else:
        print("No new tiles to download.")

if __name__ == "__main__":
    grid_file = "data/grid_500m_filtered.parquet"
    output_directory = "data/imagery/nl/rgb"
    country_code = "NL"
    
    # Start the async event loop
    asyncio.run(main(grid_file, output_directory, country_code, max_concurrent_requests=100))