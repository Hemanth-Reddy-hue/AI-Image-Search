import json
import pickle
import numpy as np
import google.generativeai as genai
import requests

# Configure Google Gemini API
API_KEY = "AIzaSyCwaqFch1pnOFIBKJ5rHZS7596jgRlbGeg"
genai.configure(api_key=API_KEY)

# GitHub Repository Base URL
GITHUB_BASE_URL = "https://github.com/Hemanth-Reddy-hue/AI-Image-Search/blob/main/data/"
CAPTION_FILE_URL = "https://raw.githubusercontent.com/Hemanth-Reddy-hue/AI-Image-Search/main/src/search/image_captions.json"
EMBEDDINGS_FILE_URL = "https://raw.githubusercontent.com/Hemanth-Reddy-hue/AI-Image-Search/main/src/search/image_caption_embeddings.pkl"

def fetch_json_from_github(url):
    """Fetch JSON data from a GitHub raw URL."""
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return response.json()
        except json.JSONDecodeError:
            print(f"❌ JSON decoding failed: {url}")
            return {}
    else:
        print(f"❌ Failed to fetch JSON from {url} (Status Code: {response.status_code})")
        return {}

def fetch_pickle_from_github(url):
    """Fetch and load a pickle file from a GitHub raw URL."""
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return pickle.loads(response.content)
        except Exception as e:
            print(f"❌ Failed to load pickle file from {url}: {e}")
            return []
    else:
        print(f"❌ Failed to fetch pickle file from {url} (Status Code: {response.status_code})")
        return []

# Load captions and embeddings from GitHub
image_captions = fetch_json_from_github(CAPTION_FILE_URL)
image_data = fetch_pickle_from_github(EMBEDDINGS_FILE_URL)

# Extract embeddings and paths
embeddings = np.array([item["embedding"] for item in image_data]).astype("float32") if image_data else np.array([])
image_paths = [item["path"] for item in image_data] if image_data else []

def generate_embedding(text):
    """Generates an embedding for a text query using the Gemini API."""
    try:
        response = genai.embed_content(
            model="text-embedding-004",
            content=text
        )
        return np.array(response["embedding"], dtype="float32")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return None

def format_github_url(original_path):
    """Converts local image paths to GitHub blob URLs for frontend display."""
    relative_path = original_path.split("data/")[-1]  # Extract after "data/"
    return f"{GITHUB_BASE_URL}{relative_path}".replace("\\", "/")  # Ensure correct format

def search_images(query, top_k=10):
    """Finds the top-K matching images for a query and returns GitHub URLs."""
    query_embedding = generate_embedding(query)
    if query_embedding is None or embeddings.size == 0:
        return []

    # Compute similarity
    query_embedding = query_embedding.reshape(1, -1)
    dot_products = np.dot(embeddings, query_embedding.T)
    indices = np.argsort(dot_products, axis=0)[-top_k:][::-1]  # Get top-k indices

    results = []
    for idx in indices.flatten():
        original_path = image_paths[idx]
        github_url = format_github_url(original_path)
        caption = image_captions.get(original_path, "No caption available")

        results.append({"path": github_url, "caption": caption, "score": float(dot_products[idx])})

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
