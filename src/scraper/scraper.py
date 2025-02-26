import os
import requests
from utils import save_image  # Import helper function for downloading images

API_KEY = "48997683-2debef01ab2a554c0d4dfb8a9"
QUERIES = ["Guns","Anime","wars"]  
PER_PAGE = 50  # Number of images per page
TOTAL_PAGES = 4  # Total number of pages
BASE_DIR = "../../data/"  # Storage directory

# Loop through each category
for query in QUERIES:
    save_dir = os.path.join(BASE_DIR, query)
    os.makedirs(save_dir, exist_ok=True)  # Ensure folder exists
    image_counter = 1  # Start counting from 1

    for page in range(1, TOTAL_PAGES + 1):
        url = f"https://pixabay.com/api/?key={API_KEY}&q={query}&image_type=photo&per_page={PER_PAGE}&page={page}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            images = data.get("hits", [])  # Extract image details
            
            for img in images:
                img_url = img["largeImageURL"]
                filename = f"{query}_{image_counter}.jpg"  # Serial naming
                print(f"saving to {os.path.join(save_dir, filename)}")
                save_image(img_url, os.path.join(save_dir, filename))
                image_counter += 1  # Increment counter
        else:
            print(f"❌ Failed to retrieve images for {query} (Page {page}) - Status Code: {response.status_code}")
