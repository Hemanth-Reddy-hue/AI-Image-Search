import faiss
import pickle
import numpy as np

# Path to embeddings file
EMBEDDINGS_FILE = "embeddings.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"

# Load embeddings from pickle file
def load_embeddings():
    """Loads stored embeddings and image paths from a pickle file."""
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    
    image_paths = [entry["path"] for entry in data]
    embeddings = np.array([entry["embedding"] for entry in data]).astype("float32")  # Ensure FAISS-compatible dtype

    return image_paths, embeddings

# Create FAISS index and store embeddings
def build_faiss_index(embeddings):
    """Builds a FAISS index for fast vector search."""
    dim = embeddings.shape[1]  # Get embedding dimension
    index = faiss.IndexFlatL2(dim)  # L2 (Euclidean) distance-based FAISS index
    index.add(embeddings)  # Add all embeddings

    return index

# Save FAISS index to file
def save_faiss_index(index):
    """Saves the FAISS index to a file."""
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"✅ FAISS index saved to {FAISS_INDEX_FILE}")

# Run FAISS processing
if __name__ == "__main__":
    print("📥 Loading embeddings...")
    image_paths, embeddings = load_embeddings()

    print("🔨 Building FAISS index...")
    faiss_index = build_faiss_index(embeddings)

    print("💾 Saving FAISS index...")
    save_faiss_index(faiss_index)

    print(f"\n✅ Stored {len(image_paths)} image embeddings in FAISS!")
