import os
import osmium
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from tqdm import tqdm

# -----------------------------------------------
# Config
# -----------------------------------------------
OSM_PBF = "data/osm_raw/netherlands-latest.osm.pbf"
ROAD_BUFFER_OUT = "data/osm_filtered/nl/nl_road_buffer.parquet"

# Cached outputs and skip extraction if already saved
BUILDINGS_OUT = "data/osm_filtered/nl/nl_buildings.parquet"
VEGETATION_OUT = "data/osm_filtered/nl/nl_vegetation.parquet"
WATER_OUT = "data/osm_filtered/nl/nl_water.parquet"

# Final output (one polygon per grid cell showing the residual parking area)
RESIDUAL_OUT = "data/pressure_estimation/nl_residual_areas.parquet"

# Grid to define the spatial extent and cell boundaries
GRID_PARQUET = "data/grid_500m_filtered.parquet"
COUNTRY_CODE = "NL"

CRS = "EPSG:28992"


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
# Step 1: Extract OSM polygon features
# -----------------------------------------------
class PolygonFeatureHandler(osmium.SimpleHandler):
    """
    Extracts closed polygon features from OSM PBF.
    accepted_tags: dict of { tag_key: set_of_values_or_None }
      - None as the value set means accept any value for that key.
      - e.g. {"building": None} accepts building=yes, building=house, etc.
      - e.g. {"landuse": {"forest", "grass"}} accepts only those two values.
    """
    # Initialize
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
            if len(outer_rings) == 1:
                poly = Polygon(outer_rings[0])
            else:
                poly = MultiPolygon([Polygon(r) for r in outer_rings])
            if poly.is_valid and not poly.is_empty:
                self.polygons.append(poly)
        except Exception:
            pass


def extract_osm_polygons(osm_pbf, accepted_tags, out_path, label, crs):
    """
    Extracts polygon features from OSM PBF matching the specified tags.
    Saves as GeoParquet and loads from cache on subsequent runs.
    """
    if os.path.exists(out_path):
        print(f"Loading existing {label} from {out_path}")
        return gpd.read_parquet(out_path)

    print(f"Extracting {label} from OSM PBF...")
    handler = PolygonFeatureHandler(accepted_tags)
    handler.apply_file(osm_pbf, locations=True, idx="flex_mem")

    if not handler.polygons:
        print(f"  No {label} features found.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_crs(crs)
    
    # WIP: make it usable for multiple countries
    gdf = gpd.GeoDataFrame(
        geometry=handler.polygons, crs="EPSG:4326"
    ).to_crs(crs)

    save_if_not_exists(gdf, out_path, label)
    print(f"  Extracted {len(gdf):,} {label} polygons.")
    return gdf


# -----------------------------------------------
# Step 2: Load all non-parking feature layers
# -----------------------------------------------
def load_non_parking_features(osm_pbf, road_buffer_path, crs):
    """
    Loads or extracts all OSM feature layers that are subtracted from the
    grid cell area to produce the residual parking area.

    Layers:
      Buildings: building=* — any mapped structure
      Vegetation: landuse and natural tags for green areas
      Water: natural=water, waterway=riverbank, landuse=reservoir etc.
      Roads: previously-made road buffers from parking_pressure.py

    Note on tag selection:
      We are intentionally conservative, as in we only subtract features that are
      absolutely non-parking. Industrial yards, construction sites, and
      untagged open spaces are NOT subtracted because they may contain parking.
      This means the residual will still overestimate parking area in some cells,
      but it is better to overestimate than to incorrectly remove real parking.
    """
    buildings = extract_osm_polygons(
        osm_pbf,
        accepted_tags={"building": None},
        out_path=BUILDINGS_OUT,
        label="buildings",
        crs=crs
    )

    vegetation = extract_osm_polygons(
        osm_pbf,
        accepted_tags={
            "landuse": {
                "forest", "grass", "meadow", "orchard", "vineyard",
                "allotments", "village_green", "recreation_ground", "cemetery"
            },
            "natural": {
                "wood", "scrub", "heath", "grassland",
                "fell", "tundra", "wetland"
            },
            "leisure": {
                "park", "garden", "nature_reserve",
                "golf_course", "pitch", "sports_centre"
            },
        },
        out_path=VEGETATION_OUT,
        label="vegetation",
        crs=crs
    )

    water = extract_osm_polygons(
        osm_pbf,
        accepted_tags={
            "natural":  {"water", "bay", "strait"},
            "waterway": {"riverbank", "dock", "boatyard"},
            "landuse":  {"reservoir", "basin"},
        },
        out_path=WATER_OUT,
        label="water bodies",
        crs=crs
    )

    print(f"Loading road buffer from {road_buffer_path}...")
    roads = gpd.read_parquet(road_buffer_path)

    return buildings, vegetation, water, roads


# -----------------------------------------------
# Step 3: Compute residual area per grid cell
# -----------------------------------------------
def compute_residual(grid, buildings, vegetation, water, roads):
    """
    For each grid cell, subtracts all non-parking feature polygons to produce
    the residual area, the space left over that is assumed to be available
    for parking.

    Process per cell:
    1. Find all features from each layer that intersect the cell (spatial index)
    2. Clip each intersecting feature to the cell boundary
    3. Union all clipped features into one mask polygon
    4. Subtract the mask from the cell polygon
    5. Store the residual geometry and its area

    The output is a GeoDataFrame with one row per grid cell, where the geometry
    is the residual polygon.

    Cell-by-cell vs single overlay
    Single overlay cause a lot of memories, so process is likely to freeze or keeps executing. Cell-by-cell would
    avoid it.
    """
    print("Computing residual area per grid cell...")

    # Combine all non-parking layers into one GeoDataFrame.
    # We keep a 'source' column so the output parquet is inspectable.
    all_features = gpd.GeoDataFrame(
        pd.concat([
            buildings[["geometry"]].assign(source="building"),
            vegetation[["geometry"]].assign(source="vegetation"),
            water[["geometry"]].assign(source="water"),
            roads[["geometry"]].assign(source="road"),
        ], ignore_index=True),
        crs=grid.crs
    )

    # Build spatial index once for the full feature set
    feature_sindex = all_features.sindex
    
    # Prepare variable to store geometry and area
    residual_geoms = []
    residual_areas = []
    cell_areas = []
    
    # Iterate over grid
    for idx, cell in tqdm(grid.iterrows(), total=len(grid),
                          desc="  Subtracting features"):
        cell_geom = cell.geometry
        cell_area = cell_geom.area

        # Query spatial index for candidate features
        candidate_idx = list(feature_sindex.intersection(cell_geom.bounds))
        candidates    = all_features.iloc[candidate_idx]

        # Filter to features that actually intersect (bounds query has false positives)
        actually_intersect = candidates[candidates.intersects(cell_geom)]

        if actually_intersect.empty:
            residual_geoms.append(cell_geom)
            residual_areas.append(cell_area)
            cell_areas.append(cell_area)
            continue

        # Clip each intersecting feature to the cell boundary and union
        clipped = actually_intersect.geometry.intersection(cell_geom)
        clipped = clipped[~clipped.is_empty & clipped.is_valid]

        if clipped.empty:
            residual_geoms.append(cell_geom)
            residual_areas.append(cell_area)
            cell_areas.append(cell_area)
            continue

        mask = union_all(clipped.values)
        residual = cell_geom.difference(mask)

        # Ensure valid geometry
        if residual.is_empty or not residual.is_valid:
            residual = cell_geom
            residual_area = cell_area
        else:
            residual_area = residual.area

        residual_geoms.append(residual)
        residual_areas.append(max(residual_area, 0.0))
        cell_areas.append(cell_area)

    result = gpd.GeoDataFrame(
        {
            "GRD_ID":           grid["GRD_ID"].values,
            "cell_area_m2":     cell_areas,
            "residual_area_m2": residual_areas,
            # Fraction of cell area that is residual. Used to see if results make sense or not
            "residual_fraction": np.array(residual_areas) / np.array(cell_areas),
            "geometry":         residual_geoms
        },
        crs=grid.crs
    )

    return result


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

    # Load or extract all non-parking OSM features
    buildings, vegetation, water, roads = load_non_parking_features(
        OSM_PBF, ROAD_BUFFER_OUT, CRS
    )

    # Compute residual area per cell
    residual_gdf = compute_residual(grid, buildings, vegetation, water, roads)

    # Save
    save_if_not_exists(residual_gdf, RESIDUAL_OUT, "residual areas")

    print("\nSummary:")
    print(residual_gdf[["cell_area_m2", "residual_area_m2",
                         "residual_fraction"]].describe())


if __name__ == "__main__":
    main()