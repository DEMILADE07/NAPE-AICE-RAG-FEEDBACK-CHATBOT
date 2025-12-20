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
    # Check if we're in a Streamlit runtime and secrets are available
    if hasattr(st, 'secrets'):
        try:
            # Try to access GOOGLE_CREDENTIALS from secrets
            # It might be a string (JSON) or already a dict
            google_creds = st.secrets.get("GOOGLE_CREDENTIALS", None)
            if google_creds:
                # Write to temp file for gspread to use
                import tempfile
                import json as json_lib
                import os
                
                # Handle both string and dict formats
                if isinstance(google_creds, str):
                    # Parse JSON string - handle both raw JSON and triple-quoted strings
                    google_creds = google_creds.strip()
                    if google_creds.startswith('"""') or google_creds.startswith("'''"):
                        # Remove triple quotes
                        google_creds = google_creds[3:-3].strip()
                    
                    # Try to parse - if it fails due to control characters, try to fix them
                    try:
                        creds_dict = json_lib.loads(google_creds)
                    except json_lib.JSONDecodeError as e:
                        error_msg = str(e).lower()
                        if 'control character' in error_msg:
                            # Try to fix newlines in private_key field
                            import re
                            # Find and fix the private_key field
                            if 'BEGIN PRIVATE KEY' in google_creds and 'END PRIVATE KEY' in google_creds:
                                # Escape newlines between BEGIN and END
                                begin_idx = google_creds.find('BEGIN PRIVATE KEY')
                                end_idx = google_creds.find('END PRIVATE KEY') + len('END PRIVATE KEY')
                                private_section = google_creds[begin_idx:end_idx]
                                private_section_fixed = private_section.replace('\n', '\\n').replace('\r', '\\r')
                                google_creds = google_creds[:begin_idx] + private_section_fixed + google_creds[end_idx:]
                                try:
                                    creds_dict = json_lib.loads(google_creds)
                                except:
                                    raise ValueError(f"Could not parse credentials JSON. Please ensure newlines in private_key are escaped as \\n")
                        else:
                            raise
                elif isinstance(google_creds, dict):
                    # Already a dict
                    creds_dict = google_creds
                else:
                    raise ValueError(f"Unexpected type for GOOGLE_CREDENTIALS: {type(google_creds)}")
                
                # Create temp file
                temp_creds = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json_lib.dump(creds_dict, temp_creds, indent=2)
                temp_creds.close()
                CREDENTIALS_PATH = Path(temp_creds.name)
                print(f"✅ Using credentials from Streamlit secrets (temp file: {CREDENTIALS_PATH})")
        except (KeyError, AttributeError, TypeError, json_lib.JSONDecodeError, ValueError) as e:
            # Secrets key doesn't exist or parsing failed - use default file
            print(f"⚠️ Could not read GOOGLE_CREDENTIALS from secrets: {e}")
            import traceback
            print(traceback.format_exc())
            pass
        else:
            if not google_creds:
                print("⚠️ GOOGLE_CREDENTIALS not found in Streamlit secrets")
except (ImportError, RuntimeError):
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
                print(f"✅ Using GROQ_API_KEY from Streamlit secrets")
            else:
                print("⚠️ GROQ_API_KEY not found in Streamlit secrets")
        except (KeyError, AttributeError, TypeError) as e:
            # Key doesn't exist or secrets not available - use env var
            print(f"⚠️ Could not read GROQ_API_KEY from secrets: {e}")
            pass
except (ImportError, RuntimeError):
    # Streamlit not available (e.g., when running pipeline.py) - use env var
    pass

# Debug: Check if GROQ_API_KEY is set (but don't print the actual key)
if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY is empty. LLM will not be available.")
else:
    print(f"✅ GROQ_API_KEY is set (length: {len(GROQ_API_KEY)} characters)")
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

