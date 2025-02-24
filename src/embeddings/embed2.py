import os
import time
import pickle
import google.generativeai as genai

# Configure Google Gemini API
API_KEY = "AIzaSyCwaqFch1pnOFIBKJ5rHZS7596jgRlbGeg"  
genai.configure(api_key=API_KEY)

# Use the embedding model
MODEL_NAME = "models/embedding-001"

# Base directory containing image folders
BASE_DIR = "../../data/"
OUTPUT_FILE = "embeddings.pkl"  # File to store embeddings

# Function to generate a caption for an image
def generate_caption(image_path):
    """Generates a simple caption based on the image filename."""
    filename = os.path.basename(image_path).replace("_", " ").split(".")[0]
    return f"This is an image of {filename}."

# Function to generate embeddings
def generate_embedding(text, title="Image Description", retries=3):
    """Generates text embeddings using the embedding-001 model."""
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model=MODEL_NAME,
                content=text,
                task_type="retrieval_document",
                title=title
            )
            if response:
                return response["embedding"]  # Ensure we extract only the embedding
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
    return None  # Return None if all attempts fail

# Recursive function to process images in all subdirectories
def process_images(directory):
    """Recursively processes images and extracts text embeddings."""
    image_embeddings = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(root, filename)

                # Generate caption for the image
                caption = generate_caption(image_path)

                # Get the embedding for the caption
                embedding = generate_embedding(caption)

                if embedding:
                    print(f"✅ Processed: {image_path}")
                    image_embeddings.append({"path": image_path, "embedding": embedding})
                else:
                    print(f"❌ Failed: {image_path}")

    return image_embeddings

# Run processing
if __name__ == "__main__":
    all_embeddings = process_images(BASE_DIR)

    # Save embeddings to a pickle file
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_embeddings, f)

    # Print all image paths with embeddings
    print("\n=== Final Image Embeddings (Stored in embeddings.pkl) ===")
    for item in all_embeddings:
        print(f"🖼 {item['path']} → {item['embedding'][:5]}... (truncated)")

    print(f"\n✅ Embeddings saved to {OUTPUT_FILE}")
