import geopandas as gpd
from requests import Request
from owslib.wfs import WebFeatureService
import os

def get_amsterdam_boundary():
    """
    Fetch Amsterdam municipality boundary from PDOK WFS using a simple GET request,
    then load into GeoDataFrame with geopandas.
    """
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ams_boundary.gpkg")

    print("Fetching Amsterdam boundary from PDOK...")
    # URL for WFS backend
    url = "https://service.pdok.nl/kadaster/brk-bestuurlijke-gebieden/wfs/v1_0"

    # Initialize
    wfs = WebFeatureService(url=url)
    
    # Define parameters
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "bestuurlijkegebieden:Gemeentegebied",
        "outputFormat": "application/json",
    }
    
    # Parse URL with parameters
    wfs_request_url = Request('GET', url, params=params).prepare().url
    
    # Read data from URL
    data = gpd.read_file(wfs_request_url)
    
    # FiLter Amsterdam
    ams_boundary = data[data["naam"] == "Amsterdam"]
    
    # Ensure projection is RD New
    ams_boundary = ams_boundary.to_crs(28992)
    
    # Save file
    ams_boundary.to_file(output_path, driver="GPKG")
    
if __name__ == "__main__":
    get_amsterdam_boundary()