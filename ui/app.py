import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/search"
IMAGE_SERVER_URL = "http://127.0.0.1:8000/images/"

def compress_and_resize_image(image_content, max_size=(800, 800)):
    """Compress and resize an image while maintaining aspect ratio."""
    try:
        img = Image.open(BytesIO(image_content))
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        return buffered.getvalue()
    except Exception:
        return None

# Set Dark Theme in Streamlit
st.set_page_config(
    page_title="AI Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Dark Mode Styling
st.markdown(
    """
    <style>
        html, body, [class*="stApp"] {
            background-color: #0e0e0e;
            color: #ffffff;
        }
        div[data-testid="stTextInput"] > div:first-child {
            max-width: 400px !important;
            margin: auto;
        }
        input::placeholder {
            color: #444 !important;
        }
        /* Dark theme for sidebar */
        section[data-testid="stSidebar"] {
            background-color: #191919 !important;
        }
        .stButton button {
            background-color: #333333 !important;
            color: white !important;
        }
        .stButton button:hover {
            background-color: #555555 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar - Search History
with st.sidebar:
    st.markdown("### 📜 Search History", unsafe_allow_html=True)
    
    # Initialize session state for history if it doesn't exist
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    # Display search history
    if st.session_state.search_history:
        for term in st.session_state.search_history:
            if st.button(f"🔍 {term}", key=f"history_{term}"):
                query = term  # Clicking history should trigger a new search
    else:
        st.markdown("<p style='color: #888;'>No search history yet.</p>", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 style='text-align: center;'>🔍 AI Image Search</h1>", unsafe_allow_html=True)
st.markdown("---")

# Search Box
query = st.text_input("", placeholder="Search for images...", key="search-box")

# Perform Search on Enter
if query:
    with st.spinner("🔍 Searching for images..."):
        try:
            response = requests.post(API_URL, json={"query": query})
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                # Store search history
                if query not in st.session_state.search_history:
                    st.session_state.search_history.insert(0, query)
                    if len(st.session_state.search_history) > 10:
                        st.session_state.search_history.pop()

                # Display Results
                if results:
                    st.markdown("### 📸 Search Results")
                    st.balloons()  # 🎈 Celebration for successful retrieval

                    cols = st.columns(3)
                    for idx, result in enumerate(results):
                        col = cols[idx % 3]
                        with col:
                            image_path = result["path"].split("Intern_Project/")[-1]
                            image_url = f"{IMAGE_SERVER_URL}{image_path.replace('\\', '/')}"

                            # Fetch and resize image
                            img_response = requests.get(image_url)
                            if img_response.status_code == 200:
                                img_data = compress_and_resize_image(img_response.content)
                                if img_data:
                                    st.image(img_data, use_container_width=True)
                else:
                    st.warning("No matching images found. Try a different search term.")
        except requests.exceptions.RequestException:
            st.error("Unable to connect to the search API. Please check if the server is running.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #aaaaaa;'>Powered by AI • Built with Streamlit</p>", unsafe_allow_html=True)
