from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Ensure `src.search` is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search.gemini_search import search_images  # Import search function

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory search history (stores last 5 unique searches)
search_history = []

class SearchQuery(BaseModel):
    query: str

@app.post("/search")
def search(query: SearchQuery):
    """Search for images based on the query using Gemini embeddings."""
    results = search_images(query.query, top_k=10)

    # Maintain last 5 unique searches
    if query.query not in search_history:
        search_history.append(query.query)
        if len(search_history) > 5:
            search_history.pop(0)  # Keep only last 5 searches

    return {
        "results": results,
        "history": search_history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))  # Dynamic port support
