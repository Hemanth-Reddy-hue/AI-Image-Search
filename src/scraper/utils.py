import requests
import os

def save_image(url, path):
    """Downloads and saves an image from a given URL."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"✅ Saved: {path}")
        else:
            print(f"⚠️ Failed to download {url} - Status Code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error saving image {url}: {e}")
