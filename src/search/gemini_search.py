import json
import pickle
import numpy as np
import google.generativeai as genai

# Configure Google Gemini API
API_KEY = "AIzaSyCwaqFch1pnOFIBKJ5rHZS7596jgRlbGeg"
genai.configure(api_key=API_KEY)

# Model Name
EMBEDDING_MODEL = "models/embedding-001"

# File Paths
CAPTION_FILE = "src/search/image_captions.json"
EMBEDDINGS_FILE = "src/search/image_caption_embeddings.pkl"

# Load captions
with open(CAPTION_FILE, "r") as f:
    image_captions = json.load(f)

# Load embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    image_data = pickle.load(f)

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
    dot_products = np.dot(np.stack([item["embedding"] for item in image_data]), query_embedding.T)
    indices = np.argsort(dot_products, axis=0)[-top_k:][::-1]  # Get top-k indices

    results = []
    for idx in indices.flatten():
        original_path = image_paths[idx]
        
        # Convert the path to match the correct format
        formatted_path = original_path.split("Intern_Project")[-1]  # Remove everything before "Intern_Project"
        formatted_path = f"C:\\Users\\reddy\\OneDrive\\Desktop\\Intern_Project{formatted_path.replace('/', '\\')}"  # Convert to Windows format
        
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
