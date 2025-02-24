from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import os

from src.search.gemini_search import search_images  # Import the search function

app = FastAPI()

# In-memory search history
search_history = []

class SearchQuery(BaseModel):
    query: str

@app.post("/search")
def search(query: SearchQuery):
    results = search_images(query.query, top_k=10)
    search_history.append(query.query)
    return {
        "results": results,
        "history": list(set(search_history[-5:]))  # Last 5 unique searches
    }

@app.get("/images/{image_path:path}")
def get_image(image_path: str):
    """Serves images via API."""
    full_path = os.path.join("data", image_path)  # Adjust path based on project structure
    if os.path.exists(full_path):
        return FileResponse(full_path)
    return {"error": "Image not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
