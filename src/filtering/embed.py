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

# Define data folder
DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tests"))

# Initialize FAISS index
embedding_dim = 512  # CLIP embedding dimension
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Store image paths
image_paths = []

def extract_embedding(image_path):
    """Extract CLIP embedding for an image."""
    try:
        with Image.open(image_path) as img:
            image = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image).cpu().numpy()

        return image_features
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None

# Process all images and store embeddings
for category in os.listdir(DATA_FOLDER):  
    category_path = os.path.join(DATA_FOLDER, category)

    if os.path.isdir(category_path):  
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            embedding = extract_embedding(img_path)
            if embedding is not None:
                faiss_index.add(embedding)
                image_paths.append(img_path)

# Save FAISS index and image paths
faiss.write_index(faiss_index, "faiss/faiss_index.bin")
with open("faiss/image_paths.txt", "w") as f:
    f.writelines("\n".join(image_paths))

print(f"✅ Stored {len(image_paths)} image embeddings in FAISS.")
