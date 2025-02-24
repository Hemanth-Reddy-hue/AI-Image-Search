import faiss
import numpy as np
import time
import pickle
import random

# Load FAISS index
INDEX_PATH = "faiss_index.bin"
EMBEDDINGS_PATH = "embeddings.pkl"

# Load stored FAISS index
index = faiss.read_index(INDEX_PATH)

# Load image embeddings and paths
with open(EMBEDDINGS_PATH, "rb") as f:
    image_embeddings = pickle.load(f)

# Convert image embeddings to NumPy array
vectors = np.array([item["embedding"] for item in image_embeddings]).astype("float32")
paths = [item["path"] for item in image_embeddings]

# 1️⃣ Check Index Size & Storage
print("=== FAISS Index Information ===")
print(f"Total vectors in FAISS index: {index.ntotal}")
print(f"Expected vectors from stored embeddings: {len(vectors)}")

# 2️⃣ Search Accuracy Test
random_idx = random.randint(0, len(vectors) - 1)
sample_vector = vectors[random_idx].reshape(1, -1)

k = 7  # Retrieve top-5 similar images
D, I = index.search(sample_vector, k)

print("\n=== Search Accuracy Test ===")
print(f"Query Image: {paths[random_idx]}")
print("Top 5 Retrieved Images:")
for i, idx in enumerate(I[0]):
    print(f"Rank {i+1}: {paths[idx]} (Distance: {D[0][i]:.4f})")

# 3️⃣ Speed Benchmarking
num_queries = 100
query_vectors = np.array(random.choices(vectors, k=num_queries)).astype("float32")
start_time = time.time()
index.search(query_vectors, k)
elapsed_time = time.time() - start_time

print("\n=== Speed Benchmarking ===")
print(f"Time taken for {num_queries} queries: {elapsed_time:.4f} sec")
print(f"Average time per query: {elapsed_time / num_queries:.6f} sec")
