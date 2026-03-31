import os
import requests
from tqdm import tqdm

# -----------------------------------------------
# Config
# -----------------------------------------------
ECW_URL  = "https://fsn1.your-objectstorage.com/hwh-ortho/2025/Ortho/LR_ecw/2025_landellijk_LRL_RGB_v2.ecw"
ECW_PATH = "data/imagery/nl/2025_landellijk_LRL_RGB_v2.ecw"

# -----------------------------------------------
# Download
# -----------------------------------------------
def download_ecw(url, path):
    # Create the folder if it doesn't exist yet
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Check if the file already exists and how many bytes are already on disk.
    # This lets us resume a partial download instead of starting over.
    existing_bytes = os.path.getsize(path) if os.path.exists(path) else 0

    if existing_bytes > 0:
        print(f"Found partial download: {existing_bytes / 1e9:.2f} GB already on disk. Resuming...")
    
    # The 'Range' header tells the server to start sending from where we left off.
    # For example: "bytes=1073741824-" means "send everything from byte 1 GB onwards".
    # If existing_bytes is 0, we just don't send this header and get the full file.
    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    response = requests.get(url, stream=True, headers=headers)

    # HTTP 206 means the server understood our Range request and is sending a partial file.
    # HTTP 200 means the server ignored our Range request and is sending the full file.
    # If we get 200 but already have some bytes, we overwrite to avoid a corrupted file.
    if response.status_code == 200 and existing_bytes > 0:
        print("Server does not support resume (returned 200 instead of 206). Restarting download.")
        existing_bytes = 0

    # Total size = what the server is sending now + what we already have on disk
    incoming_bytes = int(response.headers.get("content-length", 0))
    total_bytes    = existing_bytes + incoming_bytes

    print(f"Total file size: {total_bytes / 1e9:.2f} GB")

    # Open in append-binary mode ('ab') so we add to the existing partial file.
    # If existing_bytes is 0, 'ab' behaves the same as 'wb' (starts fresh).
    with open(path, "ab") as f, tqdm(
        total=total_bytes,
        initial=existing_bytes,      # tqdm starts its bar at the already-downloaded amount
        unit="B",
        unit_scale=True,
        desc="Downloading ECW"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # read 1 MB at a time
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"\nDone. File saved to: {path}")

if __name__ == "__main__":
    download_ecw(ECW_URL, ECW_PATH)