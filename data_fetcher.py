import requests
import pandas as pd
import os
import time

# =========================
# 1️⃣ SENTINEL HUB CREDENTIALS
# =========================
CLIENT_ID = "a933c14d-d7ae-426d-ad46-70847892c390"
CLIENT_SECRET = "9kpNmKtoZGvR32r25rOG7bpYtOio9yqJ"

# =========================
# 2️⃣ GET ACCESS TOKEN
# =========================
auth_url = "https://services.sentinel-hub.com/oauth/token"

auth_payload = {
    "grant_type": "client_credentials",
    "client_id": "a933c14d-d7ae-426d-ad46-70847892c390",
    "client_secret": "9kpNmKtoZGvR32r25rOG7bpYtOio9yqJ"
}

auth_response = requests.post(auth_url, data=auth_payload)

if auth_response.status_code != 200:
    raise Exception("Failed to get access token", auth_response.text)

ACCESS_TOKEN = auth_response.json()["access_token"]
print("Access token generated ✅")

# =========================
# 3️⃣ LOAD TRAINING DATA
# =========================
df = pd.read_csv("train(1)(train(1)).csv")
print("Training data loaded ✅")

# =========================
# 4️⃣ CREATE IMAGE DIRECTORY
# =========================
IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# =========================
# 5️⃣ FUNCTION TO FETCH IMAGE
# =========================
def fetch_sentinel_image(lat, lon, img_id):
    url = "https://services.sentinel-hub.com/api/v1/process"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "bounds": {
                "bbox": [
                    lon - 0.002, lat - 0.002,
                    lon + 0.002, lat + 0.002
                ],
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                }
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2023-01-01T00:00:00Z",
                        "to": "2023-12-31T23:59:59Z"
                    }
                }
            }]
        },
        "output": {
            "width": 224,
            "height": 224,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/png"}
            }]
        },
        "evalscript": """
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
          };
        }

        function evaluatePixel(sample) {
          return [sample.B04, sample.B03, sample.B02];
        }
        """
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        with open(f"{IMAGE_DIR}/{img_id}.png", "wb") as f:
            f.write(response.content)
        print(f"Downloaded image {img_id}")
    else:
        print(f"Failed {img_id} | Status: {response.status_code}")

# =========================
# 6️⃣ TEST WITH FEW ROWS
# =========================
import os

downloaded = 0
skipped = 0

downloaded = 0
skipped = 0

for _, row in df.iterrows():
    house_id = row["id"]
    img_path = f"{IMAGE_DIR}/{house_id}.png"

    # Skip if image already exists
    if os.path.exists(img_path):
        skipped += 1
        continue

    fetch_sentinel_image(
        row["lat"],
        row["long"],
        house_id
    )

    downloaded += 1
    time.sleep(0.5)  # rate limit

print("Download finished ✅")
print("Downloaded:", downloaded)
print("Skipped:", skipped)

