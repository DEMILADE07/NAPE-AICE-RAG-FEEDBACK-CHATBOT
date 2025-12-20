"""Configuration settings for NAPE RAG Chatbot"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
DB_PATH = BASE_DIR / "nape_feedback.db"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# Google Sheets
MASTER_SHEET_NAME = "NAPE Monitoring & Evaluation Team"  # Update if different
MASTER_SHEET_URL = ""  # Optional: if you have the direct URL

# Embedding Model (free, local)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, good quality

# LLM Settings
# Option 1: Ollama (local, free)
USE_OLLAMA = True
# Model selection: 
# - "llama3.1" - Better quality, requires ~4.2 GiB RAM, slower (~60-120s per query)
# - "llama3.2:3b" - Faster (3-5x), requires ~2-3 GiB RAM, good quality (~15-30s per query)
OLLAMA_MODEL = "llama3.2:3b"  # Faster model for better performance

# Option 2: Groq API (free tier, fast)
USE_GROQ = False
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-70b-versatile"

# RAG Settings
TOP_K_RESULTS = 10  # Number of feedback items to retrieve per query
MAX_CONTEXT_LENGTH = 2000  # Max characters for LLM context
ENABLE_CACHING = True  # Cache query results for faster responses

# Event Categories (for better organization)
EVENT_CATEGORIES = {
    "WiGE": ["WiGE", "Women in Geosciences"],
    "Technical": ["Technical Sessions", "Poster Session", "Basin Evaluation"],
    "Social": ["African Nite", "President's Night", "Awards", "Ice Breaker"],
    "Logistics": ["Accommodation", "Feeding", "Transportation", "Registration"],
    "Professional": ["Leadership Forum", "Management Session", "AGM"],
    "Competition": ["Hackathon", "Quiz", "Pitch", "Competition"],
    "Other": []
}

# All hotels available in the accommodation form (17 total)
ALL_ACCOMMODATION_HOTELS = [
    'Beni Apartments',
    'BWC Hotel',
    'Chateux De Atlantique',
    'Eko Hotel & Suites',
    'Exclusive Mansion',
    'H210',
    'Hotellnn',
    'Hotels No. 35.',
    'Morning Side',
    'Newcastle Hotel',
    'Posh Hotel',
    'Peninsula Hotel',
    'Presken Hotel',
    'S&S Hotel & Suits',
    'Timeoak Hotel',
    'VirginRose Hotel',
    'Villa Toscana'
]

