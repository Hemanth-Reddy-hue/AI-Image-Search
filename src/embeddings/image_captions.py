import os
import time
import json
import pickle
import base64

import google.generativeai as genai
from PIL import Image

# Configure Google Gemini API
API_KEY = "AIzaSyCwaqFch1pnOFIBKJ5rHZS7596jgRlbGeg"
genai.configure(api_key=API_KEY)

# Model names
EMBEDDING_MODEL = "models/embedding-001"
VISION_MODEL = "gemini-2.0-flash-exp"

# Paths
BASE_DIR = "data/"
CAPTION_FILE = "src/search/image_captions.json"
EMBEDDINGS_FILE = "src/search/embeddings.pkl"

# Function to generate captions using Gemini Vision API
def generate_caption(image_path, retries=3):
    """Generates an image caption using the Gemini API."""
    for attempt in range(retries):
        try:
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()

            # Encode image as base64
            b64_image = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_data).decode("utf-8"),
            }

            # Generate caption using Gemini API
            model = genai.GenerativeModel(VISION_MODEL)
            response = model.generate_content([b64_image, "Describe this image."])

            if hasattr(response, "text") and response.text:
                return response.text.strip()  # Extract caption
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed for {image_path}: {e}")
            time.sleep(2)

    return None  # Return None if caption generation fails

# Function to generate text embeddings
def generate_embedding(title, text, retries=3):
    """Generates text embeddings using the Gemini API."""
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document",
                title=title
            )
            if response and "embedding" in response:
                return response["embedding"]
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    return None  # Return None if all attempts fail

# Process images: Generate captions + embeddings
def process_images(directory):
    """Processes images, generates captions, and extracts embeddings."""
    image_embeddings = []
    captions = {}

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, filename)

                # Generate caption
                caption = generate_caption(image_path)
                if not caption:
                    print(f"❌ Failed to generate caption: {image_path}")
                    continue  

                # Store caption
                captions[image_path] = caption  

                # Generate embedding
                embedding = generate_embedding("Image Description", caption)
                if embedding:
                    print(f"✅ Processed: {image_path} → {caption}")
                    image_embeddings.append({"path": image_path, "embedding": embedding})
                else:
                    print(f"❌ Failed to generate embedding: {image_path}")

    return captions, image_embeddings

# Run processing
if __name__ == "__main__":
    all_captions, all_embeddings = process_images(BASE_DIR)

    # Save captions to JSON
    with open(CAPTION_FILE, "w") as f:
        json.dump(all_captions, f, indent=4)

    # Save embeddings to pickle
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(f"\n✅ Captions saved to {CAPTION_FILE}")
    print(f"✅ Embeddings saved to {EMBEDDINGS_FILE}")
