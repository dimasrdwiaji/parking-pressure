import os
import osmium
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------
# Config
# -----------------------------------------------
OSM_PBF         = "data/osm_raw/netherlands-latest.osm.pbf"
ROAD_BUFFER_OUT = "data/osm_filtered/nl/nl_road_buffer.parquet"
BUILDINGS_OUT   = "data/osm_filtered/nl/nl_buildings.parquet"
VEGETATION_OUT  = "data/osm_filtered/nl/nl_vegetation.parquet"
WATER_OUT       = "data/osm_filtered/nl/nl_water.parquet"
RESIDUAL_OUT    = "data/pressure_estimation/nl_residual_areas.parquet"

GRID_PARQUET    = "data/grid_500m_filtered.parquet"
COUNTRY_CODE    = "NL"
CRS             = "EPSG:28992"

# Road proximity: residual is clipped to within this distance of road centerlines.
# Parking must be road-accessible. Anything beyond this distance is almost
# certainly private garden, courtyard, or building interior — not parking.
ROAD_PROXIMITY_M = 15

# Parallel processing: number of grid chunks to split work across.
# Each chunk runs in its own process. Set to cpu_count() - 1 to leave
# one core free for the OS, or lower if memory is a concern.
N_WORKERS = max(1, cpu_count() - 1)

# Number of grid cells per chunk. Smaller chunks use less memory per worker
# but have more process-spawn overhead. 500 is a good balance for 135k cells.
CHUNK_SIZE = 500


# -----------------------------------------------
# Helper
# -----------------------------------------------
def save_if_not_exists(gdf, path, label="file"):
    if os.path.exists(path):
        print(f"[skip] {label} already exists at {path}")
        return
    gdf.to_parquet(path)
    print(f"Saved {len(gdf):,} rows → {path}")


# -----------------------------------------------
# Step 1: OSM extraction
# -----------------------------------------------
class PolygonFeatureHandler(osmium.SimpleHandler):
    """
    Extracts closed polygon features from OSM PBF using the area() callback.
    area() handles both simple closed ways and multipolygon relations,
    which is important for large parks, forests, and water bodies.
    """
    def __init__(self, accepted_tags):
        super().__init__()
        self.accepted_tags = accepted_tags
        self.polygons      = []

    def _matches(self, tags):
        for key, values in self.accepted_tags.items():
            val = tags.get(key)
            if val is None:
                continue
            if values is None or val in values:
                return True
        return False

    def area(self, a):
        if not self._matches(a.tags):
            return
        try:
            outer_rings = []
            for ring in a.outer_rings():
                coords = [(n.lon, n.lat) for n in ring]
                if len(coords) >= 3:
                    outer_rings.append(coords)
            if not outer_rings:
                return
            poly = (Polygon(outer_rings[0]) if len(outer_rings) == 1
                    else MultiPolygon([Polygon(r) for r in outer_rings]))
            if poly.is_valid and not poly.is_empty:
                self.polygons.append(poly)
        except Exception:
            pass


def extract_osm_polygons(osm_pbf, accepted_tags, out_path, label, crs):
    if os.path.exists(out_path):
        print(f"Loading existing {label} from {out_path}")
        return gpd.read_parquet(out_path)

    print(f"Extracting {label} from OSM PBF...")
    handler = PolygonFeatureHandler(accepted_tags)
    handler.apply_file(osm_pbf, locations=True, idx="flex_mem")

    if not handler.polygons:
        print(f"  No {label} features found.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_crs(crs)

    gdf = gpd.GeoDataFrame(geometry=handler.polygons, crs="EPSG:4326").to_crs(crs)
    save_if_not_exists(gdf, out_path, label)
    print(f"  Extracted {len(gdf):,} {label} polygons.")
    return gdf


# -----------------------------------------------
# Step 2: Build road proximity mask
# -----------------------------------------------
def build_road_proximity_mask(road_buffer_path, osm_pbf, crs, proximity_m):
    """
    Builds a polygon representing all areas within ROAD_PROXIMITY_M of any
    road centerline. This is used to clip the residual — anything outside
    this mask is assumed inaccessible and excluded from parking area.

    We need the road centerlines (not the road buffers from parking_pressure.py)
    because road buffers are for filtering cars ON roads, not for defining
    the accessible zone NEAR roads. We re-extract centerlines here at all
    road types including residential, since any road can have adjacent parking.
    """
    proximity_cache = road_buffer_path.replace(
        "nl_road_buffer.parquet", "nl_road_proximity_mask.parquet"
    )

    if os.path.exists(proximity_cache):
        print(f"Loading existing road proximity mask from {proximity_cache}")
        return gpd.read_parquet(proximity_cache)

    centerlines_cache = road_buffer_path.replace(
        "nl_road_buffer.parquet", "nl_road_centerlines.parquet"
    )

    if os.path.exists(centerlines_cache):
        print("Loading road centerlines from cache...")
        roads = gpd.read_parquet(centerlines_cache)
    else:
        print("Extracting road centerlines for proximity mask...")

        class RoadCenterlineHandler(osmium.SimpleHandler):
            def __init__(self):
                super().__init__()
                self.roads = []
                self.skip = {
                    "footway", "cycleway", "path", "steps",
                    "pedestrian", "bridleway", "track",
                    "construction", "proposed"
                }
            def way(self, w):
                highway = w.tags.get("highway")
                if not highway or highway in self.skip:
                    return
                try:
                    from shapely.geometry import LineString
                    coords = [(n.lon, n.lat) for n in w.nodes]
                    if len(coords) >= 2:
                        self.roads.append(LineString(coords))
                except Exception:
                    pass

        handler = RoadCenterlineHandler()
        handler.apply_file(osm_pbf, locations=True)

        from shapely.geometry import LineString
        roads = gpd.GeoDataFrame(
            geometry=handler.roads, crs="EPSG:4326"
        ).to_crs(crs)

    print(f"Building {proximity_m}m road proximity mask from {len(roads):,} roads...")

    # Buffer all roads by proximity_m and union into one mask.
    # We chunk the union to avoid memory issues on large road networks.
    chunk_size = 5000
    parts      = []
    for i in range(0, len(roads), chunk_size):
        chunk = roads.iloc[i : i + chunk_size]
        parts.append(unary_union(chunk.geometry.buffer(proximity_m)))

    proximity_mask = unary_union(parts)

    mask_gdf = gpd.GeoDataFrame(
        {"description": ["road_proximity_mask"]},
        geometry=[proximity_mask],
        crs=crs
    )
    save_if_not_exists(mask_gdf, proximity_cache, "road proximity mask")
    return mask_gdf


# -----------------------------------------------
# Step 3: Pre-clip all features to study area extent
# -----------------------------------------------
def clip_to_extent(gdf, extent_geom, label):
    """
    Clips a feature GeoDataFrame to the bounding box of the study area.
    This reduces the number of features processed in subsequent steps,
    especially important for road networks that may extend beyond NL borders.
    """
    print(f"  Clipping {label} to study extent...")
    clipped = gdf[gdf.intersects(extent_geom)].copy()
    print(f"  {len(gdf):,} → {len(clipped):,} {label} features after clip.")
    return clipped


# -----------------------------------------------
# Step 4: Parallel residual computation
# -----------------------------------------------
def process_chunk(args):
    """
    Worker function — processes one chunk of grid cells.
    Called by multiprocessing.Pool, so must be a top-level function
    (not a nested function or lambda) to be picklable.

    For each cell:
      1. Find intersecting non-parking features using spatial index
      2. Clip them to the cell boundary
      3. Subtract from cell to get residual
      4. Intersect residual with road proximity mask
         (removes backyards, interior courtyards, private gardens)
      5. Return cell ID, residual geometry, and area
    """
    chunk_gdf, all_features_gdf, proximity_mask_geom = args

    feature_sindex = all_features_gdf.sindex
    results        = []

    for idx, cell in chunk_gdf.iterrows():
        cell_geom = cell.geometry
        cell_area = cell_geom.area

        try:
            # Find candidates via spatial index (fast bounding box query)
            candidate_idx  = list(feature_sindex.intersection(cell_geom.bounds))
            candidates     = all_features_gdf.iloc[candidate_idx]
            intersecting   = candidates[candidates.intersects(cell_geom)]

            if intersecting.empty:
                raw_residual = cell_geom
            else:
                clipped = intersecting.geometry.intersection(cell_geom)
                clipped = clipped[~clipped.is_empty & clipped.is_valid]
                if clipped.empty:
                    raw_residual = cell_geom
                else:
                    mask         = unary_union(clipped.values)
                    raw_residual = cell_geom.difference(mask)

            # Apply road proximity constraint:
            # keep only residual that is within ROAD_PROXIMITY_M of a road.
            # This removes backyards and private land that is not road-accessible.
            if proximity_mask_geom is not None and not raw_residual.is_empty:
                accessible_residual = raw_residual.intersection(proximity_mask_geom)
            else:
                accessible_residual = raw_residual

            if accessible_residual.is_empty or not accessible_residual.is_valid:
                accessible_residual = raw_residual
                residual_area       = max(raw_residual.area, 0.0)
            else:
                residual_area = max(accessible_residual.area, 0.0)

            results.append({
                "GRD_ID":           cell.get("GRD_ID", str(idx)),
                "cell_area_m2":     cell_area,
                "residual_area_m2": residual_area,
                "residual_fraction": residual_area / cell_area if cell_area > 0 else 0.0,
                "geometry":         accessible_residual
            })

        except Exception as e:
            # On geometry error, fall back to full cell area
            results.append({
                "GRD_ID":           cell.get("GRD_ID", str(idx)),
                "cell_area_m2":     cell_area,
                "residual_area_m2": cell_area,
                "residual_fraction": 1.0,
                "geometry":         cell_geom
            })

    return results


def compute_residual_parallel(grid, all_features_gdf, proximity_mask_gdf):
    """
    Splits the grid into chunks and distributes across N_WORKERS processes.

    Why multiprocessing instead of threading?
    Python's GIL prevents true parallelism with threads for CPU-bound work.
    multiprocessing spawns separate processes each with their own GIL,
    giving true parallel execution across cores.

    The proximity mask geometry is passed as a plain Shapely object (not a
    GeoDataFrame) to avoid pickle overhead — Shapely geometries are picklable
    but GeoDataFrames carry extra metadata that slows inter-process transfer.
    """
    print(f"Computing residual areas using {N_WORKERS} workers...")

    proximity_mask_geom = proximity_mask_gdf.geometry.iloc[0]

    # Split grid into chunks
    chunks = [
        grid.iloc[i : i + CHUNK_SIZE].copy()
        for i in range(0, len(grid), CHUNK_SIZE)
    ]
    print(f"  {len(grid):,} cells split into {len(chunks)} chunks of ~{CHUNK_SIZE}")

    # Package args for each worker
    args = [(chunk, all_features_gdf, proximity_mask_geom) for chunk in chunks]

    all_results = []
    with Pool(processes=N_WORKERS) as pool:
        for chunk_result in tqdm(
            pool.imap(process_chunk, args),
            total=len(chunks),
            desc="  Processing chunks"
        ):
            all_results.extend(chunk_result)

    result_gdf = gpd.GeoDataFrame(all_results, crs=grid.crs)
    return result_gdf


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    # Load and filter grid to NL
    print("Loading grid...")
    grid = gpd.read_parquet(GRID_PARQUET)
    grid = grid[grid["CNTR_ID"].str.contains(COUNTRY_CODE)].copy()
    grid = grid.reset_index(drop=True)
    grid = grid[grid["GRD_ID"] == "N3082500E4018000"] # TRY ONLY THIS SPECIFIC GRID FIRST
    if grid.crs.to_epsg() != int(CRS.split(":")[1]):
        grid = grid.to_crs(CRS)
    print(f"Grid cells: {len(grid):,}")

    # Study area extent for pre-clipping features
    extent_geom = grid.geometry.union_all()

    # Extract OSM features
    buildings = extract_osm_polygons(
        OSM_PBF,
        {"building": None},
        BUILDINGS_OUT, "buildings", CRS
    )

    vegetation = extract_osm_polygons(
        OSM_PBF,
        {
            "landuse": {"forest", "grass", "meadow", "orchard", "vineyard",
                        "allotments", "village_green", "recreation_ground",
                        "cemetery"},
            "natural": {"wood", "scrub", "heath", "grassland",
                        "fell", "tundra", "wetland"},
            "leisure": {"park", "garden", "nature_reserve", "golf_course",
                        "pitch", "sports_centre"},
        },
        VEGETATION_OUT, "vegetation", CRS
    )

    water = extract_osm_polygons(
        OSM_PBF,
        {
            "natural":  {"water", "bay", "strait"},
            "waterway": {"riverbank", "dock", "boatyard"},
            "landuse":  {"reservoir", "basin"},
        },
        WATER_OUT, "water bodies", CRS
    )

    print("Loading road buffer...")
    roads = gpd.read_parquet(ROAD_BUFFER_OUT)

    # Build road proximity mask
    proximity_mask_gdf = build_road_proximity_mask(
        ROAD_BUFFER_OUT, OSM_PBF, CRS, ROAD_PROXIMITY_M
    )

    # Pre-clip all features to NL extent to reduce per-cell work
    print("Pre-clipping features to NL extent...")
    buildings  = clip_to_extent(buildings,  extent_geom, "buildings")
    vegetation = clip_to_extent(vegetation, extent_geom, "vegetation")
    water      = clip_to_extent(water,      extent_geom, "water")
    roads      = clip_to_extent(roads,      extent_geom, "road buffers")

    # Merge all non-parking features into one GeoDataFrame
    all_features = gpd.GeoDataFrame(
        pd.concat([
            buildings[["geometry"]].assign(source="building"),
            vegetation[["geometry"]].assign(source="vegetation"),
            water[["geometry"]].assign(source="water"),
            roads[["geometry"]].assign(source="road"),
        ], ignore_index=True),
        crs=CRS
    )
    print(f"Total non-parking features: {len(all_features):,}")

    # Compute residual in parallel
    residual_gdf = compute_residual_parallel(grid, all_features, proximity_mask_gdf)

    # Save
    save_if_not_exists(residual_gdf, RESIDUAL_OUT, "residual areas")

    print("\nSummary:")
    print(residual_gdf[[
        "cell_area_m2", "residual_area_m2", "residual_fraction"
    ]].describe())


if __name__ == "__main__":
    main()