"""Configuration settings for NAPE RAG Chatbot"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "nape_feedback.db"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# Credentials - Support both file and Streamlit secrets
CREDENTIALS_PATH = BASE_DIR / "credentials.json"  # Default

# Try to get from Streamlit secrets (for Streamlit Cloud)
# Only works when running in Streamlit context
try:
    import streamlit as st
    # Check if we're actually in a Streamlit runtime
    if hasattr(st, 'secrets') and hasattr(st.secrets, '_file_path'):
        try:
            GOOGLE_CREDENTIALS_JSON = st.secrets.get("GOOGLE_CREDENTIALS", None)
            if GOOGLE_CREDENTIALS_JSON:
                # Write to temp file for gspread to use
                import tempfile
                import json as json_lib
                temp_creds = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json_lib.dump(json_lib.loads(GOOGLE_CREDENTIALS_JSON), temp_creds)
                temp_creds.close()
                CREDENTIALS_PATH = Path(temp_creds.name)
        except (FileNotFoundError, KeyError, AttributeError):
            # Secrets file not found or key doesn't exist - use default
            pass
except (ImportError, AttributeError, RuntimeError, FileNotFoundError):
    # Streamlit not available or not in Streamlit context (e.g., when running pipeline.py)
    # Use default file path
    pass

# Google Sheets
MASTER_SHEET_NAME = "NAPE Monitoring & Evaluation Team"  # Update if different
MASTER_SHEET_URL = ""  # Optional: if you have the direct URL

# Embedding Model (free, local)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, good quality

# LLM Settings
# Option 1: Ollama (local, free) - Only works locally, not on Streamlit Cloud
USE_OLLAMA = False  # Set to False for Streamlit Cloud deployment
# Model selection: 
# - "llama3.1" - Better quality, requires ~4.2 GiB RAM, slower (~60-120s per query)
# - "llama3.2:3b" - Faster (3-5x), requires ~2-3 GiB RAM, good quality (~15-30s per query)
OLLAMA_MODEL = "llama3.2:3b"  # Faster model for better performance

# Option 2: Groq API (free tier, fast) - Recommended for Streamlit Cloud
USE_GROQ = True  # Set to True for Streamlit Cloud deployment
# Get API key from environment variable or Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Try to get from Streamlit secrets (for Streamlit Cloud)
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        try:
            # Try to access secrets - this will work in Streamlit Cloud
            secret_key = st.secrets.get("GROQ_API_KEY", None)
            if secret_key:
                GROQ_API_KEY = secret_key
        except (KeyError, AttributeError, TypeError):
            # Key doesn't exist or secrets not available - use env var
            pass
except (ImportError, RuntimeError):
    # Streamlit not available (e.g., when running pipeline.py) - use env var
    pass
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

