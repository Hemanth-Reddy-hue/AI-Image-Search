# AI-Powered Image Search & Web Scraping Project

## 📌 Project Overview
This project is a **semantic image search engine** that allows users to search for images based on natural language queries. It integrates **web scraping, image processing, embeddings, vector search, and an AI-powered UI** to deliver accurate image search results.

### 🚀 Key Features
- **Web Scraping**: Scrapes images from the web using `BeautifulSoup` and `Scrapy`.
- **Image Captioning**: Generates captions for images.
- **Embeddings & Vector Search**: Uses **Google Gemini API** to generate text embeddings and **FAISS** for efficient similarity search.
- **LLM-powered Retrieval**: Uses RAG-based AI retrieval for search queries.
- **FastAPI Backend**: Handles search requests and serves image data.
- **Streamlit UI**: Provides a user-friendly interface for searching images.
- **Search History**: Displays recent searches for better UX.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Create a Virtual Environment (Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up API Keys
- Obtain a **Google Gemini API key** from Google AI.
- Obtain a **Pixabay API Key** for webscraping
- Create a `.env` file and add:
  ```sh
  GEMINI_API_KEY=your_api_key_here
  ```

---

## 🚀 Running the Application

### 1️⃣ Start the FastAPI Backend
```sh
python src/api/main.py
```
This runs the API server at `http://127.0.0.1:8000/`

### 2️⃣ Start the Streamlit UI
```sh
streamlit run ui/app.py
```
This launches the user interface.

---

## 🔍 How It Works
1. **Scrape & Store Images**: Images and metadata are scraped and stored.
2. **Generate Embeddings**: Captions are converted into embeddings via Google Gemini.
3. **Indexing with FAISS**: Images are indexed in FAISS for fast retrieval.
4. **Semantic Search**: When a user searches, the system finds the closest matching images.
5. **Display Results**: The UI fetches and displays search results along with history.

---

## 🛠️ Project Structure
```
Intern_Project/
│── src/
│   ├── scraper/         # Web scraping logic
│   ├── filtering/       # Image classification & filtering
│   ├── embeddings/      # Embedding generation & FAISS indexing
│   ├── search/          # Search functionality (RAG + FAISS)
│   ├── api/             # FastAPI backend server
│── ui/                  # Streamlit frontend
│── requirements.txt     # Dependencies
│── README.md            # Documentation
```

---

## 📌 API Endpoints (FastAPI)
| Method | Endpoint   | Description |
|--------|-----------|-------------|
| `POST` | `/search` | Search images based on query |

---

## 🎉 Future Enhancements
- Improve **UI styling** and responsiveness.
- Optimize **embedding model performance**.
- Implement **user authentication** for personalized searches.

---

## 👨‍💻 Author
Developed by **Hemanth Reddy**.

---

## 📜 License
This project is licensed under the **MIT License**.

---

🔹 **Happy Searching!** 🔍🎉

