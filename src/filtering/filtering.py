import clip
import torch
import faiss
import os
from PIL import Image
from model import load_clip_model
import numpy as np

# Load CLIP model
model, preprocess = load_clip_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define FAISS index and image paths
FAISS_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "faiss/faiss_index.bin"))
IMAGE_PATHS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "faiss/image_paths.txt"))

# Ensure FAISS index exists
if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(IMAGE_PATHS_FILE):
    print("❌ FAISS index or image paths file not found. Make sure to run the embedding generation first.")
    exit(1)

# Load FAISS index
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Load stored image paths
with open(IMAGE_PATHS_FILE, "r") as f:
    stored_image_paths = f.read().splitlines()

# Define test folder
TEST_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

def get_embedding(image_path):
    """Extract CLIP embedding for a test image."""
    try:
        with Image.open(image_path) as img:
            image = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image).cpu().numpy()

        return image_features.astype(np.float32)  # Ensure correct FAISS dtype
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None

def is_ad_image(image_path, threshold=0.25):
    """Check if an image is an ad using FAISS similarity search."""
    embedding = get_embedding(image_path)
    if embedding is None:
        return False

    # Ensure embedding has the correct shape for FAISS
    embedding = np.ascontiguousarray(embedding)  # Fix FAISS shape issues

    # Search for nearest match in FAISS
    distances, _ = faiss_index.search(embedding, k=1)
    min_distance = distances[0][0]

    return min_distance < threshold  # If similarity is high, classify as ad.

def scan_images(folder):
    """Recursively scan and filter images in all subdirectories."""
    for root, _, files in os.walk(folder):  # Recursively get all files
        for img_name in files:
            img_path = os.path.join(root, img_name)

            if is_ad_image(img_path):
                print(f"🛑 Removing Ad Image: {img_path}")
                # os.remove(img_path)  # Uncomment to enable deletion
            else:
                print(f"✅ Keeping Image: {img_path}")

# Start scanning recursively
scan_images(TEST_FOLDER)

print("✅ FAISS filtering completed.")
