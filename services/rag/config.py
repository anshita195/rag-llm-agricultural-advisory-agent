"""Shared RAG configuration."""

import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
CHROMA_PATH = "services/rag/chroma_db"
DB_PATH = "data/agrisage.db"
COLLECTION_NAME = "agri"

AUTHORITATIVE_SOURCES = {
    "weather_forecast",
    "soil_card",
    "market_prices",
    "enam_trades",
    "real_weather_data",
    "real_mandi_prices",
    "OpenWeatherMap",
    "SoilGrids_ISRIC",
}
MIN_PROVENANCE_SCORE = 0.6
ACTIONABLE_KEYWORDS = ["irrigate", "spray", "apply", "plant", "harvest", "fertilize", "dose", "timing"]

SOURCE_URLS = {
    "weather_forecast": "https://mausam.imd.gov.in",
    "soil_card": "https://soilhealth.dac.gov.in",
    "market_prices": "https://agmarknet.gov.in",
    "enam_trades": "https://enam.gov.in",
    "OpenWeatherMap": "https://openweathermap.org",
    "SoilGrids_ISRIC": "https://soilgrids.org",
}

ALLOWED_DISTRICTS = {"roorkee", "haridwar"}

# Places outside eval coverage — explicit detection in queries
OUT_OF_REGION_PLACES = {
    "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "kolkata",
    "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "kanpur",
}

DANGEROUS_KEYWORDS = [
    "pesticide",
    "insecticide",
    "fungicide",
    "herbicide",
    "chemical",
    "spray",
    "dose",
    "dosage",
    "poison",
]
