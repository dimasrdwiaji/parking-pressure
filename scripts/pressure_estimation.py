import os
import osmium
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# -----------------------------------------------
# Config
# -----------------------------------------------
# Input car detections (GeoParquet, EPSG:28992)
CAR_PARQUET = "data/imagery/nl/pred/detected_cars_final.parquet"

# OSM filtered PBF for the country (from your existing pipeline)
OSM_PBF = "data/osm_raw/netherlands-latest.osm.pbf"

# Output paths
ROAD_CENTERLINES_OUT = "data/osm_filtered/nl_road_centerlines.parquet"
ROAD_BUFFER_OUT      = "data/osm_filtered/nl_road_buffer.parquet"
CLUSTERS_OUT         = "data/nl_parking_clusters.parquet"

# CRS to work in — must be metric and match car parquet
CRS = "EPSG:28992"

# DBSCAN params
DBSCAN_EPS         = 10    # max gap (meters) between cars in same cluster
DBSCAN_MIN_SAMPLES = 2     # minimum cars to form a cluster

# Corner rounding: buffer out then back in by this amount (meters)
ROUND_BUFFER = 2.0

# Buffer distances in meters by road type.
# Wider roads get larger buffers because driving lanes are further from centerline.
# We keep residential/service small to preserve legitimate street-side parking.
ROAD_BUFFERS = {
    "motorway":       15,
    "motorway_link":  15,
    "trunk":          12,
    "trunk_link":     12,
    "primary":        10,
    "primary_link":   10,
    "secondary":       8,
    "secondary_link":  8,
    "tertiary":        6,
    "tertiary_link":   6,
    "residential":     4,
    "service":         4,
    "living_street":   3,
    "unclassified":    5,
}
DEFAULT_ROAD_BUFFER = 5  # for any road type not listed above


# -----------------------------------------------
# Step 1: Extract road centerlines from OSM PBF
# -----------------------------------------------
class RoadHandler(osmium.SimpleHandler):
    """
    Extracts road ways from OSM PBF.
    We store each way as a list of (lon, lat) node locations
    plus the highway tag so we can apply type-specific buffers later.
    osmium requires locations=True when calling apply_file so that
    node coordinates are available on each way.
    """
    def __init__(self):
        super().__init__()
        self.roads = []  # list of { "highway": str, "coords": [(lon, lat), ...] }

    def way(self, w):
        highway = w.tags.get("highway")
        if highway is None:
            return
        # Only keep road types we want to buffer
        # Footways, cycleways, paths etc. are excluded
        if highway not in ROAD_BUFFERS and highway != "unclassified":
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
            if len(coords) >= 2:
                self.roads.append({"highway": highway, "coords": coords})
        except osmium.InvalidLocationError:
            # Some nodes may have missing coordinates — skip those ways
            pass


def extract_roads(osm_pbf, out_path, crs):
    """
    Extracts road centerlines from the OSM PBF, saves as GeoParquet.
    If the file already exists, loads and returns it directly.
    """
    if os.path.exists(out_path):
        print(f"Loading existing road centerlines from {out_path}")
        return gpd.read_parquet(out_path)

    print("Extracting road centerlines from OSM PBF...")
    handler = RoadHandler()
    handler.apply_file(osm_pbf, locations=True)

    from shapely.geometry import LineString
    geometries = []
    highway_types = []

    for road in handler.roads:
        line = LineString(road["coords"])
        geometries.append(line)
        highway_types.append(road["highway"])

    gdf = gpd.GeoDataFrame(
        {"highway": highway_types},
        geometry=geometries,
        crs="EPSG:4326"   # OSM coordinates are always WGS84
    )

    # Reproject to target metric CRS for accurate buffering in meters
    gdf = gdf.to_crs(crs)
    gdf.to_parquet(out_path)
    print(f"Saved {len(gdf):,} road centerlines to {out_path}")
    return gdf


# -----------------------------------------------
# Step 2: Buffer roads by type and merge
# -----------------------------------------------
def build_road_buffer(roads_gdf, out_path):
    """
    Buffers each road by its type-specific distance, merges all into one
    unified polygon, and saves as GeoParquet.
    If already saved, loads and returns it directly.

    We save the buffer as a single-row GeoDataFrame containing the merged
    MultiPolygon. This is the mask we use to filter out driving cars.
    """
    if os.path.exists(out_path):
        print(f"Loading existing road buffer from {out_path}")
        return gpd.read_parquet(out_path)

    print("Buffering roads by type...")
    buffered_parts = []

    for highway_type, group in tqdm(
        roads_gdf.groupby("highway"), desc="Buffering road types"
    ):
        dist = ROAD_BUFFERS.get(highway_type, DEFAULT_ROAD_BUFFER)
        buffered = group.geometry.buffer(dist)
        buffered_parts.append(buffered)

    # Merge all buffered parts into one unified polygon
    print("Merging road buffers...")
    all_buffered = pd.concat(buffered_parts)
    merged = unary_union(all_buffered)

    buffer_gdf = gpd.GeoDataFrame(
        {"description": ["road_buffer"]},
        geometry=[merged],
        crs=roads_gdf.crs
    )
    buffer_gdf.to_parquet(out_path)
    print(f"Saved merged road buffer to {out_path}")
    return buffer_gdf


# -----------------------------------------------
# Step 3: Filter cars on roads
# -----------------------------------------------
def filter_road_cars(cars_gdf, road_buffer_gdf):
    """
    Removes car detections that fall within the road buffer polygon.
    These are cars driving on roads rather than parked cars.

    sjoin with predicate 'within' returns only cars inside the buffer polygon.
    We then keep the complement — cars NOT in that join result.
    """
    print(f"Cars before road filtering: {len(cars_gdf):,}")

    road_mask = road_buffer_gdf.geometry.unary_union

    # Boolean mask: True if car is within road buffer
    on_road = cars_gdf.geometry.within(road_mask)
    parked   = cars_gdf[~on_road].copy()

    print(f"Cars after road filtering:  {len(parked):,}")
    print(f"Removed {on_road.sum():,} cars on roads.")
    return parked


# -----------------------------------------------
# Step 4: DBSCAN clustering
# -----------------------------------------------
def cluster_cars(cars_gdf):
    """
    Clusters parked car centroids using DBSCAN.

    eps=10m: cars within 10m of each other are in the same cluster.
             This bridges the gap across a typical driving lane (5–6m)
             while separating genuinely distinct lots.
    min_samples=2: at least 2 cars needed to form a cluster.
                   Single isolated detections get label -1 (noise) and
                   are excluded from cluster geometry calculation.

    Returns the input GeoDataFrame with a 'cluster_id' column added.
    Noise points (cluster_id == -1) are kept in the output for reference
    but excluded from the geometry step.
    """
    print("Clustering cars with DBSCAN...")
    coords = np.array([[g.x, g.y] for g in cars_gdf.geometry])

    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        n_jobs=-1
    ).fit(coords)

    cars_gdf = cars_gdf.copy()
    cars_gdf["cluster_id"] = clustering.labels_

    n_clusters = (clustering.labels_ >= 0).sum()
    n_noise    = (clustering.labels_ == -1).sum()
    print(f"Found {len(set(clustering.labels_)) - 1:,} clusters, "
          f"{n_noise:,} noise points excluded.")
    return cars_gdf


# -----------------------------------------------
# Step 5: Build rounded rotated bounding box per cluster
# -----------------------------------------------
def build_cluster_geometries(cars_gdf):
    """
    For each cluster, computes:
      - minimum_rotated_rectangle: tightest rectangle aligned to the dominant
        car orientation within the cluster. This reflects the boxy/rectangular
        nature of parking lots better than a convex hull.
      - Rounded corners: buffer outward by ROUND_BUFFER with round join style,
        then shrink back. This softens sharp corners to better match real lot shapes.
      - car_count: number of detected cars in the cluster
      - area_m2: area of the rounded bounding box in square meters
      - density: cars per m² within the cluster area

    Density is our parking pressure metric. A higher density means more cars
    per unit area, indicating higher occupancy relative to the physical space.
    The theoretical maximum for a real lot (with driving lanes) is ~0.04 cars/m².
    We express density as a fraction of this maximum for interpretability.
    """
    print("Building cluster geometries...")

    # Only process actual clusters, not noise
    clustered = cars_gdf[cars_gdf["cluster_id"] >= 0]
    records   = []

    for cluster_id, group in tqdm(
        clustered.groupby("cluster_id"), desc="Building geometries"
    ):
        points     = MultiPoint(list(group.geometry))
        car_count  = len(group)

        # Minimum rotated rectangle — aligned to dominant car orientation
        rect       = points.minimum_rotated_rectangle

        # Round the corners: buffer out with round joins, then shrink back
        rounded    = rect.buffer(ROUND_BUFFER, join_style="round").buffer(-ROUND_BUFFER)

        area_m2    = rounded.area
        if area_m2 == 0:
            continue

        density           = car_count / area_m2
        theoretical_max   = 0.04   # cars/m² for a real lot with driving lanes
        pressure_ratio    = min(density / theoretical_max, 1.0)
        # Capped at 1.0 — values above theoretical max indicate very dense
        # on-street parking or detection errors

        records.append({
            "cluster_id":     cluster_id,
            "car_count":      car_count,
            "area_m2":        round(area_m2, 2),
            "density":        round(density, 6),
            "pressure_ratio": round(pressure_ratio, 4),
            "geometry":       rounded
        })

    result_gdf = gpd.GeoDataFrame(records, crs=cars_gdf.crs)
    print(f"Built {len(result_gdf):,} cluster geometries.")
    return result_gdf


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    # Step 1: Extract road centerlines (or load from cache)
    roads_gdf = extract_roads(OSM_PBF, ROAD_CENTERLINES_OUT, CRS)

    # Step 2: Buffer roads (or load from cache)
    road_buffer_gdf = build_road_buffer(roads_gdf, ROAD_BUFFER_OUT)

    # Step 3: Load car detections and reproject to match road CRS
    print("Loading car detections...")
    cars_gdf = gpd.read_parquet(CAR_PARQUET)
    if cars_gdf.crs.to_epsg() != int(CRS.split(":")[1]):
        cars_gdf = cars_gdf.to_crs(CRS)
    print(f"Loaded {len(cars_gdf):,} car detections.")

    # Step 4: Remove cars on roads
    parked_gdf = filter_road_cars(cars_gdf, road_buffer_gdf)

    # Step 5: Cluster remaining parked cars
    clustered_gdf = cluster_cars(parked_gdf)

    # Step 6: Build rounded rotated bounding boxes with density
    clusters_gdf = build_cluster_geometries(clustered_gdf)

    # Step 7: Save
    clusters_gdf.to_parquet(CLUSTERS_OUT)
    print(f"\nSaved {len(clusters_gdf):,} parking clusters to {CLUSTERS_OUT}")
    print(clusters_gdf[["car_count", "area_m2", "density", "pressure_ratio"]].describe())


if __name__ == "__main__":
    main()