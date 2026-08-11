#!/usr/bin/env python3
"""Core AgriSage RAG pipeline — shared by Streamlit and FastAPI."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rules_engine.fallback import get_fallback_response, safety_check, escalation_response
from rag.config import (
    ACTIONABLE_KEYWORDS,
    ALLOWED_DISTRICTS,
    AUTHORITATIVE_SOURCES,
    CHROMA_PATH,
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    MIN_PROVENANCE_SCORE,
    OUT_OF_REGION_PLACES,
    SOURCE_URLS,
)
from rag.prompts import PROMPT_TEMPLATE

project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

logger = logging.getLogger(__name__)

LLM_LOG_FILE = Path("logs/llm_requests.jsonl")
LLM_LOG_FILE.parent.mkdir(exist_ok=True)

_chroma_client = None
_collection = None
_sentence_model = None
_gemini_api_key: Optional[str] = None
_initialized = False


def initialize() -> None:
    """Load models and connections once."""
    global _chroma_client, _collection, _sentence_model, _gemini_api_key, _initialized

    if _initialized:
        return

    logger.info("Loading sentence transformer...")
    _sentence_model = SentenceTransformer(EMBEDDING_MODEL)

    chroma_path = Path(CHROMA_PATH)
    if not chroma_path.exists():
        raise FileNotFoundError(
            "Chroma database not found. Run: python -m services.rag.build_index"
        )

    _chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    _collection = _chroma_client.get_collection(COLLECTION_NAME)
    _gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not _gemini_api_key:
        logger.warning("GEMINI_API_KEY not found in environment")

    _initialized = True
    logger.info("AgriSage pipeline initialized")


def get_context_from_db(location: Optional[str] = None) -> Dict:
    """Get additional context from database based on location."""
    context: Dict = {}

    try:
        db_path = Path(DB_PATH)
        if not db_path.exists():
            return context

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if location:
            cursor.execute(
                """
                SELECT precip_prob, max_temp, min_temp, NULL as soil_moisture
                FROM reliable_weather w
                WHERE w.district LIKE ?
                ORDER BY w.date DESC LIMIT 1
                """,
                (f"%{location}%",),
            )
            result = cursor.fetchone()
            if result:
                context.update(
                    {
                        "precip_prob": result[0],
                        "max_temp": result[1],
                        "min_temp": result[2],
                        "soil_moisture": result[3],
                    }
                )

        conn.close()
    except Exception as exc:
        logger.error("Error getting context from DB: %s", exc)

    return context


def get_query_intent(query: str) -> Dict[str, float]:
    """Classify query intent and extract keywords."""
    query_lower = query.lower()

    intent_patterns = {
        "irrigation": ["irrigat", "water", "watering", "moisture", "dry", "wet"],
        "weather": ["weather", "rain", "temperature", "forecast", "climate"],
        "soil": ["soil", "ph", "nitrogen", "phosphorus", "potassium", "nutrient"],
        "market": ["price", "market", "sell", "buy", "mandi", "cost"],
        "fertilizer": ["fertiliz", "nutrient", "npk", "urea", "compost"],
        "pest": ["pest", "insect", "disease", "spray", "chemical"],
    }

    intent_scores = {}
    for intent, keywords in intent_patterns.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            intent_scores[intent] = score / len(keywords)

    return intent_scores


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma L2 distance to a bounded 0-1 similarity score."""
    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


def _metadata_relevance_score(meta: Dict, relevant_types: List[str], location: Optional[str]) -> float:
    """Bounded metadata relevance score in [0, 1]. Not vector similarity."""
    score = 0.5
    if meta.get("type") in relevant_types:
        score += 0.3
    district = meta.get("district", "")
    if location and district.lower() == location.lower():
        score += 0.2
    elif location and location.lower() in district.lower():
        score += 0.1
    return min(1.0, score)


def filter_by_metadata(
    documents: List[str],
    metadatas: List[Dict],
    distances: List[float],
    query: str,
    location: Optional[str] = None,
) -> Tuple[List[str], List[Dict], List[float]]:
    """Filter documents by metadata relevance and combine with vector similarity."""
    intent_scores = get_query_intent(query)

    if not intent_scores:
        return documents, metadatas, [_distance_to_similarity(d) for d in distances]

    primary_intent = max(intent_scores.keys(), key=lambda key: intent_scores[key])

    intent_to_type = {
        "irrigation": ["weather", "soil"],
        "weather": ["weather"],
        "soil": ["soil"],
        "market": ["market", "trade"],
        "fertilizer": ["soil"],
        "pest": ["weather", "soil"],
    }

    relevant_types = intent_to_type.get(primary_intent, [])

    filtered_docs: List[str] = []
    filtered_metas: List[Dict] = []
    relevance_scores: List[float] = []

    for doc, meta, distance in zip(documents, metadatas, distances):
        metadata_score = _metadata_relevance_score(meta, relevant_types, location)
        vector_score = _distance_to_similarity(distance)
        combined_score = min(1.0, (0.4 * vector_score) + (0.6 * metadata_score))

        if metadata_score >= 0.6:
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            relevance_scores.append(combined_score)

    return filtered_docs, filtered_metas, relevance_scores


def retrieve_documents(
    query: str, k: int = 5, location: Optional[str] = None
) -> Tuple[List[str], List[Dict], float]:
    """Hybrid retrieval: vector similarity + metadata filtering."""
    initialize()

    try:
        query_embedding = _sentence_model.encode([query]).tolist()
        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=min(k * 3, 15),
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not documents:
            return [], [], 0.0

        filtered_docs, filtered_metas, relevance_scores = filter_by_metadata(
            documents, metadatas, distances, query, location
        )

        if not filtered_docs:
            intent_scores = get_query_intent(query)
            primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "unknown"
            logger.warning(
                "No relevant data found for query: %s (intent: %s)", query, primary_intent
            )
            return [], [], 0.0

        final_docs = filtered_docs[:k]
        final_metas = filtered_metas[:k]
        final_scores = relevance_scores[:k]
        avg_retrieval_score = min(
            1.0,
            sum(final_scores) / len(final_scores) if final_scores else 0.0,
        )

        logger.info(
            "Retrieved %s filtered documents, avg score: %.3f",
            len(final_docs),
            avg_retrieval_score,
        )
        return final_docs, final_metas, avg_retrieval_score

    except Exception as exc:
        logger.error("Error retrieving documents: %s", exc)
        return [], [], 0.0


def safety_gate_check(
    query: str,
    metadatas: List[Dict],
    retrieval_score: float,
    llm_confidence: float,
) -> Dict:
    """Safety gate to prevent harmful advice without proper provenance."""
    query_lower = query.lower()
    is_actionable_query = any(keyword in query_lower for keyword in ACTIONABLE_KEYWORDS)

    if not is_actionable_query:
        return {"safe": True, "actionable": False, "gate_reason": None}

    has_authoritative_source = any(
        meta.get("source", "") in AUTHORITATIVE_SOURCES for meta in metadatas
    )
    meets_score_threshold = retrieval_score >= MIN_PROVENANCE_SCORE
    combined_confidence = 0.6 * retrieval_score + 0.4 * llm_confidence
    meets_confidence_threshold = combined_confidence >= 0.5

    if has_authoritative_source and meets_score_threshold and meets_confidence_threshold:
        return {"safe": True, "actionable": True, "gate_reason": None}

    reasons = []
    if not has_authoritative_source:
        reasons.append("no authoritative sources")
    if not meets_score_threshold:
        reasons.append(
            f"low retrieval score ({retrieval_score:.2f} < {MIN_PROVENANCE_SCORE})"
        )
    if not meets_confidence_threshold:
        reasons.append(f"low combined confidence ({combined_confidence:.2f} < 0.5)")

    return {
        "safe": False,
        "actionable": True,
        "gate_reason": "; ".join(reasons),
    }


def format_confidence_level(confidence: float) -> str:
    """Convert numeric confidence to human-readable level."""
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.5:
        return "Medium"
    return "Low"


def create_conservative_response(gate_reason: str) -> str:
    """Create conservative response when safety gate blocks actionable advice."""
    return f"""⚠️ **Insufficient authoritative data for actionable advice**

Your question appears to require specific agricultural guidance, but the available data doesn't meet our safety standards ({gate_reason}).

**Recommended actions:**
• Consult your local agricultural extension officer
• Visit the nearest Krishi Vigyan Kendra (KVK)
• Contact district agricultural department
• Speak with experienced farmers in your area

**Why we're being cautious:** Agricultural advice can significantly impact crop yields and farmer livelihoods. We only provide actionable recommendations when backed by authoritative government data sources."""


def log_llm_request(
    request_id: str,
    prompt: str,
    response: dict,
    status_code: int,
    latency: float,
    error: Optional[str] = None,
) -> None:
    """Log LLM request for debugging and monitoring."""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "model": GEMINI_MODEL,
            "prompt_length": len(prompt),
            "status_code": status_code,
            "latency_ms": round(latency * 1000, 2),
            "success": status_code == 200,
            "error": error,
            "response_tokens": response.get("usageMetadata", {}).get("totalTokenCount", 0)
            if response
            else 0,
        }
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry) + "\n")
    except Exception as exc:
        logger.error("Failed to log LLM request: %s", exc)


def call_gemini_llm(prompt: str) -> Tuple[Optional[str], float]:
    """Call Google Gemini LLM and return response with confidence."""
    initialize()

    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()

    try:
        if not _gemini_api_key:
            logger.warning("Gemini API key not available")
            return None, 0.0

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={_gemini_api_key}"
        )
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are AgriSage, an AI agricultural advisor for Indian farmers. "
                                "Always end your response with a confidence score between 0.0 and 1.0.\n\n"
                                f"{prompt}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500,
                "topP": 0.8,
                "topK": 10,
            },
        }

        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data, timeout=30)
        latency = (datetime.now() - start_time).total_seconds()

        if response.status_code == 200:
            result = response.json()
            log_llm_request(request_id, prompt, result, response.status_code, latency)

            if result.get("candidates"):
                answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                llm_confidence = 0.7
                if "confidence:" in answer.lower():
                    try:
                        conf_part = answer.lower().split("confidence:")[-1].strip()
                        conf_num = float(conf_part.split()[0])
                        if 0.0 <= conf_num <= 1.0:
                            llm_confidence = conf_num
                    except ValueError:
                        pass

                logger.info(
                    "LLM success [%s]: %.0fms, confidence: %s",
                    request_id,
                    latency * 1000,
                    llm_confidence,
                )
                return answer, llm_confidence

            log_llm_request(request_id, prompt, result, response.status_code, latency, "No candidates")
            return None, 0.0

        error_msg = f"HTTP {response.status_code}: {response.text}"
        logger.error("Gemini API error [%s]: %s", request_id, error_msg)
        log_llm_request(request_id, prompt, {}, response.status_code, latency, error_msg)
        return None, 0.0

    except Exception as exc:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = str(exc)
        logger.error("Error calling Gemini LLM [%s]: %s", request_id, error_msg)
        log_llm_request(request_id, prompt, {}, 0, latency, error_msg)
        return None, 0.0


def _build_provenance(metadatas: List[Dict], documents: List[str]) -> List[Dict]:
    provenance = []
    for meta, doc in zip(metadatas[:3], documents[:3]):
        prov_entry = {
            "source": meta["source"],
            "row_id": meta["row_id"],
            "content": doc[:200] + "..." if len(doc) > 200 else doc,
            "date": meta.get("date", "Unknown"),
            "district": meta.get("district", "Unknown"),
        }
        if meta["source"] in SOURCE_URLS:
            prov_entry["url"] = SOURCE_URLS[meta["source"]]
        provenance.append(prov_entry)
    return provenance


def _extract_places(text: str) -> set:
    """Find known place names mentioned in query text."""
    lowered = text.lower()
    found = set()
    for place in OUT_OF_REGION_PLACES | ALLOWED_DISTRICTS:
        if place in lowered:
            found.add(place)
    return found


def check_geographic_coverage(question: str, location: Optional[str] = None) -> Optional[Dict]:
    """Return an out-of-region response if the query is outside coverage."""
    mentioned = _extract_places(question)
    if location:
        mentioned.add(location.lower())

    if mentioned & OUT_OF_REGION_PLACES:
        return {
            "answer": (
                "AgriSage currently covers Roorkee and Haridwar only. "
                "I don't have weather or soil data for that location."
            ),
            "confidence": 0.2,
            "provenance": [],
            "escalate": False,
            "fallback_used": False,
            "actionable": False,
            "safety_gate": "outside_coverage_area",
        }

    if mentioned and not mentioned <= ALLOWED_DISTRICTS:
        return {
            "answer": (
                "AgriSage currently covers Roorkee and Haridwar only. "
                "I don't have verified data for the location you asked about."
            ),
            "confidence": 0.2,
            "provenance": [],
            "escalate": False,
            "fallback_used": False,
            "actionable": False,
            "safety_gate": "outside_coverage_area",
        }

    if location and location.lower() not in ALLOWED_DISTRICTS:
        return {
            "answer": (
                "AgriSage currently covers Roorkee and Haridwar only. "
                "Please select Roorkee or Haridwar as your location."
            ),
            "confidence": 0.2,
            "provenance": [],
            "escalate": False,
            "fallback_used": False,
            "actionable": False,
            "safety_gate": "outside_coverage_area",
        }

    return None


def ask(
    question: str,
    location: Optional[str] = None,
    user_id: str = "anonymous",
) -> Dict:
    """Main RAG entry point. Returns a response dict for UI and API layers."""
    del user_id  # reserved for future session tracking

    geo_block = check_geographic_coverage(question, location)
    if geo_block:
        return geo_block

    if safety_check(question):
        esc = escalation_response()
        return {
            "answer": esc["advice"],
            "confidence": esc["confidence"],
            "provenance": [],
            "escalate": True,
            "fallback_used": False,
            "actionable": False,
            "safety_gate": None,
        }

    documents, metadatas, retrieval_score = retrieve_documents(question, location=location)

    if not documents:
        context = get_context_from_db(location)
        fallback_result = get_fallback_response(question, context)
        return {
            "answer": fallback_result["advice"],
            "confidence": fallback_result["confidence"],
            "provenance": [],
            "escalate": fallback_result.get("escalate", False),
            "fallback_used": True,
            "actionable": False,
            "safety_gate": None,
        }

    context_text = "\n\n".join(
        f"Source: {meta['source']} (ID: {meta['row_id']})\nContent: {doc}"
        for doc, meta in zip(documents, metadatas)
    )
    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        question=question,
        location=location or "Not specified",
    )

    llm_response, llm_confidence = call_gemini_llm(prompt)

    if not llm_response:
        context = get_context_from_db(location)
        fallback_result = get_fallback_response(question, context)
        return {
            "answer": fallback_result["advice"],
            "confidence": fallback_result["confidence"],
            "provenance": [],
            "escalate": fallback_result.get("escalate", False),
            "fallback_used": True,
            "actionable": False,
            "safety_gate": None,
        }

    combined_confidence = 0.6 * retrieval_score + 0.4 * llm_confidence
    gate = safety_gate_check(question, metadatas, retrieval_score, llm_confidence)

    if not gate["safe"]:
        return {
            "answer": create_conservative_response(gate["gate_reason"]),
            "confidence": combined_confidence,
            "provenance": _build_provenance(metadatas, documents),
            "escalate": True,
            "fallback_used": False,
            "actionable": gate["actionable"],
            "safety_gate": gate["gate_reason"],
        }

    if combined_confidence < 0.4 or "ESCALATE" in llm_response:
        context = get_context_from_db(location)
        fallback_result = get_fallback_response(question, context)
        return {
            "answer": fallback_result["advice"],
            "confidence": fallback_result["confidence"],
            "provenance": [],
            "escalate": True,
            "fallback_used": True,
            "actionable": gate["actionable"],
            "safety_gate": None,
        }

    provenance = _build_provenance(metadatas, documents)
    unique_sources = list({entry["source"] for entry in provenance})
    enhanced_answer = (
        f"{llm_response}\n\n**Sources:** {', '.join(unique_sources)}\n"
        f"**Confidence:** {format_confidence_level(combined_confidence)}\n"
        f"**Actionability:** {'Yes' if gate['actionable'] else 'No'}"
    )

    return {
        "answer": enhanced_answer,
        "confidence": combined_confidence,
        "provenance": provenance,
        "escalate": False,
        "fallback_used": False,
        "actionable": gate["actionable"],
        "safety_gate": None,
    }


def get_health() -> Dict:
    """Return pipeline health information."""
    initialize()

    db_records = 0
    try:
        db_path = Path(DB_PATH)
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM reliable_weather")
            weather_rows = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM reliable_soil")
            soil_rows = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM reliable_markets")
            market_rows = cursor.fetchone()[0]
            db_records = weather_rows + soil_rows + market_rows
            conn.close()
    except Exception as exc:
        logger.error("Health check DB error: %s", exc)

    return {
        "status": "healthy",
        "chroma_connected": _collection is not None,
        "model_loaded": _sentence_model is not None,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "database_records": db_records,
        "vector_documents": _collection.count() if _collection else 0,
    }
