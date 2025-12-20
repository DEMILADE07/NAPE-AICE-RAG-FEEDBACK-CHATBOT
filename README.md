# NAPE 43rd AICE Feedback Analysis RAG Chatbot

AI-powered feedback analysis system for the Nigerian Association of Petroleum Explorationists (NAPE) 43rd Annual International Conference & Exhibition.

## Features

- 🤖 **Natural Language Queries**: Ask questions about conference feedback in plain English
- 📊 **Analytics Dashboard**: Visual insights into ratings, response counts, and trends
- 🔍 **Semantic Search**: Find relevant feedback using AI-powered search
- 📈 **Structured Analytics**: Quantitative analysis of ratings and statistics
- 🌐 **Web Interface**: Accessible via Streamlit Cloud (shareable link)

## Setup

### 1. Prerequisites

- Python 3.9 or higher
- Google account with access to the NAPE feedback forms
- (Optional) Ollama installed locally for free LLM, or Groq API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Google Service Account Setup

Follow the detailed instructions in `SETUP_GUIDE.md` to:
- Create a Google Cloud project
- Enable Google Sheets API
- Create a service account
- Download credentials.json
- Share your Google Sheets with the service account

### 4. (Optional) Set Up LLM

**Option A: Ollama (Recommended - Free, Local)**
```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.1
```

**Option B: Groq API (Free Tier, Fast)**
- Get API key from https://console.groq.com
- Set environment variable: `export GROQ_API_KEY=your_key`
- Update `config.py` to set `USE_GROQ = True`

### 5. Run Data Pipeline

First, ingest and process all feedback data:

```bash
python pipeline.py
```

This will:
- Connect to Google Sheets
- Collect responses from all 33 forms
- Process and structure the data
- Store in SQLite and ChromaDB

### 6. Launch Web Interface

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Query Interface

Ask natural language questions like:
- "What did attendees say about WiGE?"
- "What was the average rating for accommodation?"
- "What were common complaints about transportation?"
- "Which technical session received the best feedback?"

### Analytics Dashboard

View:
- Response counts by event
- Average ratings visualization
- Event categories breakdown
- Overall statistics

### Data Refresh

**Note:** The data refresh functionality has been removed from the UI for production safety. To update data, run:
```bash
python pipeline.py
```
This ensures proper rate limiting and prevents accidental API quota issues.

## Project Structure

```
.
├── app.py                 # Streamlit web interface
├── pipeline.py            # Data ingestion pipeline
├── data_ingestion.py      # Google Sheets connector
├── data_processor.py      # Data processing & structuring
├── storage.py             # SQLite + ChromaDB storage
├── rag_engine.py          # RAG query engine
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── SETUP_GUIDE.md         # Google Service Account setup
└── README.md              # This file
```

## Deployment

### Quick Start

1. **Run Data Pipeline First:**
   ```bash
   python pipeline.py
   ```
   This collects all 1,606 responses from Google Sheets and sets up the database.

2. **Start the Application:**
   ```bash
   streamlit run app.py
   ```

### Production Deployment

For detailed deployment instructions, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

**Quick Options:**
- **Streamlit Cloud** (Free, Easy) - See DEPLOYMENT.md for steps
- **Local Server/VPS** - Full control, can run Ollama locally
- **Docker** - Containerized deployment

**Important:** Always run `python pipeline.py` before deploying to ensure data is loaded!

## Notes

- All feedback data is anonymized (no personal information collected)
- Data is stored locally in SQLite and ChromaDB
- Embeddings are generated using free, open-source models
- No external API costs (unless using Groq API)

## Troubleshooting

**"credentials.json not found"**
- Follow SETUP_GUIDE.md to create the service account

**"Ollama not available"**
- Install Ollama from https://ollama.ai
- Run `ollama pull llama3.1`

**"No responses collected"**
- Ensure Google Sheets are shared with the service account email
- Check that form links in master sheet are correct

## License

Internal use for NAPE Monitoring & Evaluation Committee

