# AI Agricultural Assistant

A production-ready RAG (Retrieval-Augmented Generation) system that provides agricultural advice to Indian farmers using real-time data from authoritative sources.

## Demo

<img width="1901" height="1014" alt="image" src="https://github.com/user-attachments/assets/3b0e6530-0c71-41d7-811e-4ee6585792c8" />
<img width="1918" height="1015" alt="image" src="https://github.com/user-attachments/assets/cb4e05d4-d623-43c3-9305-50e21c011300" />
<img width="1917" height="1014" alt="image" src="https://github.com/user-attachments/assets/71982ecd-9b44-4f31-addf-8437f8c7aa64" />
<img width="1917" height="1016" alt="image" src="https://github.com/user-attachments/assets/9782c659-a3ba-439b-9301-ad7978eb3fef" />

## Video Demo

https://youtu.be/hJqTQ70vMiw?si=79HKOQ_kyFOCRLrO&t=62

## What It Does

AgriSage answers agricultural questions using:
- **Real weather data** from OpenWeatherMap API
- **Real soil data** from SoilGrids ISRIC API  
- **AI-powered responses** via Google Gemini 2.0
- **Safety mechanisms** that escalate complex queries to human experts

## Architecture

- **Backend**: FastAPI with SQLite database
- **Vector Search**: ChromaDB with sentence-transformer embeddings
- **Frontend**: Streamlit chat interface
- **LLM**: Google Gemini 2.0
- **Data Sources**: OpenWeatherMap, SoilGrids ISRIC, NASA POWER

## Current Data Coverage

- **Weather**: real-time forecasts
- **Soil**: pH, nutrients, composition

## Quick Start

### 1. Setup Environment
```bash
git clone <your-repo-url>
cd rag-llm-agricultural-advisory-agent
pip install -r requirements.txt
```

### 2. Configure API Keys
Create `.env` file with:
```env
OPENWEATHER_API_KEY=your_openweather_key
DATA_GOV_IN_API_KEY=your_datagovin_key
GEMINI_API_KEY=your_gemini_key
DATABASE_URL=sqlite:///data/agrisage.db
```

### 3. Initialize Data
```bash
# Fetch real data from APIs
python -m services.ingestion.reliable_api_fetcher

# Build vector search index
python -m services.rag.build_index
```

### 4. Start Application
```bash
# Terminal 1: Start API server
uvicorn services.api.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start frontend
streamlit run frontend/streamlit_app.py --server.port 8501
```

### 5. Access Application
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs (optional — Streamlit calls the RAG pipeline in-process)

## Deploy (Streamlit Community Cloud)

1. **Push all commits** — remote `main` must include `data/agrisage.db`, `services/rag/chroma_db/`, and `services/rag/pipeline.py` (6 commits ahead of origin as of eval freeze).
2. **Main file**: `frontend/streamlit_app.py` (calls `pipeline.ask()` directly — no `:8000` backend required).
3. **Python version**: In deploy **Advanced settings → Python version**, select **3.11** or **3.12**. Community Cloud defaults to **3.12**; `runtime.txt` is **not** used. Do **not** select 3.13 — `numpy`/ML wheels may fail (same as local fresh-venv test on 3.13).
4. **Secrets** (TOML in Advanced settings):
   ```toml
   GEMINI_API_KEY = "your_key"
   OPENWEATHER_API_KEY = "your_key"
   ```
5. After deploy, verify sidebar shows database records and vector docs (pre-built artifacts load from repo).

## What Works Well

### Weather Queries
- "Weather forecast for next 3 days"
- "Will it rain tomorrow in Roorkee?"
- **Source**: OpenWeatherMap API

### Soil Queries  
- "What is the soil pH in my area?"
- "Is the soil suitable for maize?"
- **Source**: SoilGrids ISRIC API

### Safety Mechanisms
- "Best time to plant mustard" → Escalates to human expert
- "Can I mix urea and DAP together?" → Escalates to human expert
- Escalation rules were expanded to cover known gaps (planting timing, fertilizer mixing) **prior to eval**; pre-eval spot-checks on those queries are engineering validation, not blind confirmation

## Current Limitations

### Geographic Coverage
- **Current**: Roorkee, Haridwar region
- **Expansion**: Add more districts to `reliable_api_fetcher.py`

## Technical Details

### Data Pipeline
1. **Ingestion**: `services/ingestion/reliable_api_fetcher.py`
2. **Vector Index**: `services/rag/build_index.py`
3. **RAG Pipeline**: `services/api/app.py`
4. **Safety Rules**: `services/rules_engine/fallback.py`

### Key Features
- **Source Attribution**: Shows which API provided the data
- **Confidence Scoring**: High/Medium/Low based on data quality
- **Provenance Tracking**: Links responses to specific data records
- **Safety Escalation**: Complex queries routed to human experts

## Project Structure

```
rag-llm-agricultural-advisory-agent/
├── services/
│   ├── api/           # FastAPI backend
│   ├── ingestion/     # Data fetching
│   ├── rag/          # Vector search
│   └── rules_engine/  # Safety mechanisms
├── frontend/         # Streamlit UI
├── data/            # SQLite database
├── logs/            # Request logs
└── scripts/         # Utility scripts
```

## Development

### Adding New Data Sources
1. Add fetcher in `services/ingestion/reliable_api_fetcher.py`
2. Update `services/rag/build_index.py` to include new data
3. Rebuild vector index: `python -m services.rag.build_index`

### Testing Changes
```bash
# Test API endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","question":"weather forecast","location":"Roorkee"}'
```

## Future Enhancements

- **Geographic Expansion**: Add more Indian districts
- **Multilingual**: Hindi language support
- **SMS Integration**: Twilio-based SMS queries


## Performance

Measured on a **23-query manual eval** (`logs/eval_23_results.json`, frozen at commit `0c67ef6`, valid run with 13s LLM spacing). Escalation rules were expanded for known gaps (planting timing, fertilizer mixing) **prior to eval**; pre-eval spot-checks on those queries are engineering validation, not blind confirmation.

| Metric | Result | Notes |
|--------|--------|-------|
| Escalation recall | 7/8 (87.5%) | #13 wheat-sowing prep answered instead of escalating — pattern gap (`prepare soil for` vs `prepare soil in`); frozen-code finding |
| Answer correctness | 22/23 (95.7%) | Same #13 miss; all other queries behaved as expected |
| Groundedness (non-escalate) | 15/15 (100%) | Weather/soil answers used retrieved data on clean LLM path |
| Retrieval precision | 21/22 (95.5%) | #23 retrieved in-region chunks before geo guard blocked |
| Strict High label match | **1/10** | See confidence calibration below |
| Geo guard (Mumbai) | Pass | Low confidence, `outside_coverage` |

### Confidence calibration (not raw underperformance)

Combined confidence = `0.6 × retrieval_score + 0.4 × llm_confidence`. When the LLM does not embed a parseable score, `llm_confidence` defaults to **0.7**. For typical good retrieval (~0.85), that yields **~0.79** — labeled **Medium** because High requires **≥ 0.8**. High is only reachable with retrieval **> ~0.92** or an explicit LLM confidence **≥ ~0.95**.

The eval's strict-High expectations were written from the README's unverified "90%+" claim, **not** from this formula. **1/10 is therefore a calibration mismatch between stated labels and threshold math**, not evidence that weather/soil answers failed — on this run, 14/14 LLM-bound weather/soil queries returned grounded direct answers. In practice, users will see mostly Medium on good answers because the High bar is nearly unreachable by construction; a post-eval improvement would recalibrate (e.g. lower High to **≥ 0.75**, or adjust the 0.6/0.4 weight split) — documented here only; not changed during the code freeze.

- **Response Time**: < 3 seconds for most queries
- **Data Freshness**: Weather updated every 3 hours
- **Vector Search**: Sub-second retrieval
- **Concurrent Users**: Tested up to 50 simultaneous requests

## Live

https://rag-llm-agricultural-advisory-agent.streamlit.app/

---
