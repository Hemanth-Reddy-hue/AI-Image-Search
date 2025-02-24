import json
import pickle
import numpy as np
import google.generativeai as genai
import os

API_KEY ="AIzaSyCwaqFch1pnOFIBKJ5rHZS7596jgRlbGeg"
if not API_KEY:
    raise ValueError("❌ Google Gemini API key is missing. Set GEMINI_API_KEY as an environment variable.")

# Configure Google Gemini API
genai.configure(api_key=API_KEY)

# Model Name
EMBEDDING_MODEL = "models/embedding-001"

# Get script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths (Relative to script)
CAPTION_FILE = os.path.join(BASE_DIR, "image_captions.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "image_caption_embeddings.pkl")

# Load captions
try:
    with open(CAPTION_FILE, "r") as f:
        image_captions = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Caption file not found: {CAPTION_FILE}")

# Load embeddings
try:
    with open(EMBEDDINGS_FILE, "rb") as f:
        image_data = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Embeddings file not found: {EMBEDDINGS_FILE}")

# Extract embeddings and paths
embeddings = np.array([item["embedding"] for item in image_data]).astype("float32")
image_paths = [item["path"] for item in image_data]

def generate_embedding(text):
    """Generates an embedding for a text query using the Gemini API."""
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        if response and "embedding" in response:
            return np.array(response["embedding"]).astype("float32")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
    return None

def search_images(query, top_k=10):
    """Finds the top-K matching images for a query and formats paths correctly."""
    query_embedding = generate_embedding(query)
    if query_embedding is None:
        return []

    # Reshape for search
    query_embedding = query_embedding.reshape(1, -1)
    dot_products = np.dot(embeddings, query_embedding.T).flatten()  # Correct shape for similarity

    # Get top-k indices
    indices = np.argsort(dot_products)[-top_k:][::-1]

    results = []
    for idx in indices:
        original_path = image_paths[idx]
        
        # Convert to OS-compatible path
        formatted_path = os.path.normpath(os.path.join(os.getcwd(), original_path.lstrip("/")))

        caption = image_captions.get(original_path, "No caption available")
        results.append({"path": formatted_path, "caption": caption, "score": float(dot_products[idx])})

    return results


# Example Usage
if __name__ == "__main__":
    query = input("\n🔍 Enter search query: ")
    results = search_images(query)

    if results:
        print("\n🎯 Top 10 Results:")
        for res in results:
            print(f"📌 {res['path']} - {res['caption']} (Score: {res['score']:.2f})")
    else:
        print("❌ No matching images found.")
