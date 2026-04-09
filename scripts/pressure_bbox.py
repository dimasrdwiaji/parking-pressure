import os
import osmium
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint, LineString
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# -----------------------------------------------
# Config
# -----------------------------------------------
CAR_PARQUET = "data/imagery/nl/pred/detected_cars_final.parquet"
OSM_PBF = "data/osm_raw/netherlands-latest.osm.pbf"

ROAD_CENTERLINES_OUT = "data/osm_filtered/nl_road_centerlines.parquet"
ROAD_BUFFER_OUT = "data/osm_filtered/nl_road_buffer.parquet"
CLUSTERS_OUT = "data/imagery/nl/nl_parking_clusters.parquet"

CRS = "EPSG:28992"

DBSCAN_EPS         = 10   # meters
DBSCAN_MIN_SAMPLES = 2
ROUND_BUFFER       = 2.0  # meters for corner rounding

# Half of a standard car width (2.4m / 2 = 1.2m).
# Car centroids are points. Create a buffer for the car in case there are only one line of parking
# which, by default, would only create a thin line, so buffer them.
CENTROID_BUFFER_M = 1.2

# Theoretical maximum density: 1 car per car footprint (2.4m × 4.8m = 11.52m2).
# Dimension is a bit iffy as there's no standardzied car footprint. 
# This is the tight packing limit with no driving lanes
# because we compute density against the bounding box of detected cars only,
# not a full lot area that includes lanes.
THEORETICAL_MAX_DENSITY = 1 / 11.52  # i think this is about 0.0868 cars/m²

# Road buffer distance
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
DEFAULT_ROAD_BUFFER = 5


# -----------------------------------------------
# Helper: save GeoDataFrame only if file does not exist
# -----------------------------------------------
def save_if_not_exists(gdf, path, label="file"):
    if os.path.exists(path):
        print(f"[skip] {label} already exists at {path}")
        return
    gdf.to_parquet(path)
    print(f"Saved {len(gdf):,} rows: {path}")


# -----------------------------------------------
# Step 1: Extract road centerlines from OSM PBF
# -----------------------------------------------
class RoadHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.roads = []

    def way(self, w):
        highway = w.tags.get("highway")
        if highway is None:
            return
        # Skip tiny roads
        skip = {
            "footway", "cycleway", "path", "steps", "pedestrian",
            "bridleway", "track", "construction", "proposed"
        }
        if highway in skip:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
            if len(coords) >= 2:
                self.roads.append({"highway": highway, "coords": coords})
        except osmium.InvalidLocationError:
            pass


def extract_roads(osm_pbf, out_path, crs):
    if os.path.exists(out_path):
        print(f"Loading existing road centerlines from {out_path}")
        return gpd.read_parquet(out_path)

    print("Extracting road centerlines from OSM PBF...")
    handler = RoadHandler()
    handler.apply_file(osm_pbf, locations=True)

    gdf = gpd.GeoDataFrame(
        {"highway": [r["highway"] for r in handler.roads]},
        geometry=[LineString(r["coords"]) for r in handler.roads],
        crs="EPSG:4326"
    ).to_crs(crs)

    save_if_not_exists(gdf, out_path, "road centerlines")
    return gdf


# -----------------------------------------------
# Step 2: Buffer roads. Keep as individual geometries 
# Union just make processing too long and stuck
# -----------------------------------------------
def build_road_buffer(roads_gdf, out_path):
    if os.path.exists(out_path):
        print(f"Loading existing road buffer from {out_path}")
        return gpd.read_parquet(out_path)

    print("Buffering roads by type...")
    buffered_rows = []

    for highway_type, group in tqdm(
        roads_gdf.groupby("highway"), desc="Buffering road types"
    ):
        dist = ROAD_BUFFERS.get(highway_type, DEFAULT_ROAD_BUFFER)
        buffered = group.copy()
        buffered["geometry"] = group.geometry.buffer(dist)
        buffered_rows.append(buffered)

    buffer_gdf = gpd.GeoDataFrame(
        pd.concat(buffered_rows, ignore_index=True),
        crs=roads_gdf.crs
    )

    save_if_not_exists(buffer_gdf, out_path, "road buffer")
    return buffer_gdf


# -----------------------------------------------
# Step 3: Filter cars on driving roads using sjoin
# -----------------------------------------------
def filter_road_cars(cars_gdf, road_buffer_gdf):
    print(f"Cars before filtering: {len(cars_gdf):,}")

    joined = gpd.sjoin(
        cars_gdf.reset_index(names="car_idx"),
        road_buffer_gdf[["geometry"]],
        how="inner",
        predicate="within"
    )

    on_road_idx = set(joined["car_idx"].unique())
    parked = cars_gdf[~cars_gdf.index.isin(on_road_idx)].copy()

    print(f"Cars after road filtering:  {len(parked):,}")
    print(f"Removed {len(on_road_idx):,} cars on roads.")
    return parked


# -----------------------------------------------
# Step 4: DBSCAN clustering
# -----------------------------------------------
def cluster_cars(cars_gdf):
    print("Clustering cars with DBSCAN...")
    coords = np.array([[g.x, g.y] for g in cars_gdf.geometry])

    clustering = DBSCAN(
        eps=DBSCAN_EPS, # There's not really a good way of capturing the cluster, won't be perfect
        min_samples=DBSCAN_MIN_SAMPLES,
        n_jobs=-1
    ).fit(coords)

    cars_gdf = cars_gdf.copy()
    cars_gdf["cluster_id"] = clustering.labels_

    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_noise    = (clustering.labels_ == -1).sum()
    print(f"Found {n_clusters:,} clusters, {n_noise:,} noise points excluded.")
    return cars_gdf


# -----------------------------------------------
# Step 5: Build rounded rotated bounding box per cluster
# -----------------------------------------------
def build_cluster_geometries(cars_gdf):
    """
    For each cluster:

    1. Buffer each car centroid by CENTROID_BUFFER_M (half car width) before
       computing the bounding rectangle. This prevents goofy near-zero-width
       rectangles when cars are parked in single line adjacent.
       Without this, minimum_rotated_rectangle on a MultiPoint produces a line-like
       rectangle with near-zero area, making density too large.

    2. Compute minimum_rotated_rectangle on the buffered geometry. This gives a
       rectangle aligned to the majority car orientation. Aim is to follow parking lots pattern.

    3. Round the rectangle corners (buffer out then back in).

    4. Compute density as cars / m². The theoretical maximum is 1 car per car
       footprint (1 / 11.52 ≈ 0.087 cars/m²), not the 0.04 figure which assumed
       driving lanes were included in the area. Here we measure against the tight
       bounding box of detected cars only, so the tighter maximum is correct.
    """
    print("Building cluster geometries...")
    clustered = cars_gdf[cars_gdf["cluster_id"] >= 0]
    records = []

    for cluster_id, group in tqdm(
        clustered.groupby("cluster_id"), desc="Building geometries"
    ):
        car_count = len(group)
        points = MultiPoint(list(group.geometry))

        # Buffer centroids by half car width before computing rectangle.
        # unary_union merges overlapping circles into one polygon so that
        # minimum_rotated_rectangle sees a shape with realistic physical width.
        buffered = points.buffer(CENTROID_BUFFER_M)
        if hasattr(buffered, '__iter__'):
            buffered = unary_union(list(buffered))

        rect = buffered.minimum_rotated_rectangle
        rounded = rect.buffer(ROUND_BUFFER, join_style="round").buffer(-ROUND_BUFFER)

        area_m2 = rounded.area
        if area_m2 == 0:
            continue

        density = car_count / area_m2
        pressure_ratio = min(density / THEORETICAL_MAX_DENSITY, 1.0)

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
    roads_gdf = extract_roads(OSM_PBF, ROAD_CENTERLINES_OUT, CRS)
    road_buffer_gdf = build_road_buffer(roads_gdf, ROAD_BUFFER_OUT)

    print("Loading car detections...")
    cars_gdf = gpd.read_parquet(CAR_PARQUET)
    if cars_gdf.crs.to_epsg() != int(CRS.split(":")[1]):
        cars_gdf = cars_gdf.to_crs(CRS)
    print(f"Loaded {len(cars_gdf):,} car detections.")

    parked_gdf = filter_road_cars(cars_gdf, road_buffer_gdf)
    clustered_gdf = cluster_cars(parked_gdf)
    clusters_gdf  = build_cluster_geometries(clustered_gdf)

    save_if_not_exists(clusters_gdf, CLUSTERS_OUT, "parking clusters")

    print(f"\nSummary:")
    print(clusters_gdf[["car_count", "area_m2", "density", "pressure_ratio"]].describe())


if __name__ == "__main__":
    main()