import os
import time
import pickle
import google.generativeai as genai

# =========================
# Configure Gemini API
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Use embedding model (NOTE: embedding-001 is deprecated but kept as requested)
MODEL_NAME = "models/embedding-001"

# Base directory containing image folders
BASE_DIR = "../../data/"
OUTPUT_FILE = "embeddings.pkl"

# =========================
# Generate caption
# =========================
def generate_caption(image_path):
    """Generates a simple caption based on the image filename."""
    filename = os.path.basename(image_path).replace("_", " ").split(".")[0]
    return f"This is an image of {filename}."

# =========================
# Generate embeddings
# =========================
def generate_embedding(text, title="Image Description", retries=3):
    """Generates text embeddings using Gemini."""
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model=MODEL_NAME,
                content=text,
                task_type="retrieval_document",
                title=title
            )
            if response and "embedding" in response:
                return response["embedding"]
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return None

# =========================
# Process images recursively
# =========================
def process_images(directory):
    image_embeddings = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(root, filename)

                caption = generate_caption(image_path)
                embedding = generate_embedding(caption)

                if embedding:
                    print(f"✅ Processed: {image_path}")
                    image_embeddings.append({
                        "path": image_path,
                        "embedding": embedding
                    })
                else:
                    print(f"❌ Failed: {image_path}")

    return image_embeddings

# =========================
# Main
# =========================
if __name__ == "__main__":
    all_embeddings = process_images(BASE_DIR)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_embeddings, f)

    print("\n=== Final Image Embeddings ===")
    for item in all_embeddings:
        print(f"🖼 {item['path']} → {item['embedding'][:5]}...")

    print(f"\n✅ Embeddings saved to {OUTPUT_FILE}")
