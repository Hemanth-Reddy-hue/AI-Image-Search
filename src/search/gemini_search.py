import os
import time
import pickle
import numpy as np
import google.generativeai as genai

# =========================
# Configure Gemini API
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/embedding-001"  # kept as-is
BASE_DIR = "../../data/"
OUTPUT_FILE = "embeddings.pkl"

# =========================
# Caption generator
# =========================
def generate_caption(image_path):
    filename = os.path.basename(image_path).replace("_", " ").split(".")[0]
    return f"This is an image of {filename}."

# =========================
# Embedding generator
# =========================
def generate_embedding(text, title="Image Description", retries=3):
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model=MODEL_NAME,
                content=text,
                task_type="retrieval_document",
                title=title
            )
            if response and "embedding" in response:
                return np.array(response["embedding"], dtype="float32")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return None

# =========================
# Process images
# =========================
def process_images(directory):
    image_embeddings = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(root, filename)
                caption = generate_caption(image_path)
                embedding = generate_embedding(caption)

                if embedding is not None:
                    image_embeddings.append({
                        "path": image_path,
                        "embedding": embedding
                    })
                    print(f"✅ Processed: {image_path}")
                else:
                    print(f"❌ Failed: {image_path}")

    return image_embeddings

# =========================
# Cosine similarity
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# Threshold-based search
# =========================
def search_images(query, threshold=0.75):
    if not os.path.exists(OUTPUT_FILE):
        print("❌ embeddings.pkl not found")
        return []

    with open(OUTPUT_FILE, "rb") as f:
        data = pickle.load(f)

    query_embedding = generate_embedding(query)
    if query_embedding is None:
        return []

    results = []
    for item in data:
        score = cosine_similarity(query_embedding, item["embedding"])
        if score >= threshold:
            results.append({
                "path": item["path"],
                "score": float(score)
            })

    # Sort by similarity descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# =========================
# Main
# =========================
if __name__ == "__main__":

    # Step 1: Build embeddings (run once)
    if not os.path.exists(OUTPUT_FILE):
        embeddings = process_images(BASE_DIR)
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"\n✅ Embeddings saved to {OUTPUT_FILE}")

    # Step 2: Search
    query = input("\n🔍 Enter search query: ")
    threshold = float(input("🎯 Enter similarity threshold (e.g. 0.75): "))

    results = search_images(query, threshold)

    if results:
        print("\n🖼 Matching Images:")
        for r in results:
            print(f"{r['path']} → similarity: {r['score']:.3f}")
    else:
        print("❌ No images matched the threshold.")
