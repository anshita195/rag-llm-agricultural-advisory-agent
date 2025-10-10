#!/usr/bin/env python3
"""
FastAPI server for AgriSage RAG system
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os
from pathlib import Path
import json
from typing import Optional, List, Dict
import sqlite3
from dotenv import load_dotenv
import logging
from datetime import datetime
import uuid
import re

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM request logging
LLM_LOG_FILE = Path("logs/llm_requests.jsonl")
LLM_LOG_FILE.parent.mkdir(exist_ok=True)

# Import fallback rules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from rules_engine.fallback import get_fallback_response, safety_check
from rag.prompts import PROMPT_TEMPLATE

app = FastAPI(title="AgriSage API", description="Agricultural Advisory RAG System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
chroma_client = None
collection = None
sentence_model = None
gemini_api_key = None

# Safety gate configuration
AUTHORITATIVE_SOURCES = {'weather_forecast', 'soil_card', 'market_prices', 'enam_trades', 'real_weather_data', 'real_mandi_prices'}
MIN_PROVENANCE_SCORE = 0.6
ACTIONABLE_KEYWORDS = ['irrigate', 'spray', 'apply', 'plant', 'harvest', 'fertilize', 'dose', 'timing']

class QueryRequest(BaseModel):
    user_id: str
    question: str
    location: Optional[str] = None
    locale: Optional[str] = "en"

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    provenance: List[Dict]
    escalate: Optional[bool] = False
    fallback_used: Optional[bool] = False
    actionable: Optional[bool] = False
    safety_gate: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on startup"""
    global chroma_client, collection, sentence_model, gemini_api_key
    
    try:
        # Initialize sentence transformer
        print("Loading sentence transformer...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Chroma client
        chroma_path = Path("services/rag/chroma_db")
        if not chroma_path.exists():
            raise FileNotFoundError("Chroma database not found. Run: python services/rag/build_index.py")
        
        chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        collection = chroma_client.get_collection("agri")
        print("Chroma database loaded")
        
        # Initialize Gemini API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            print("Gemini API key loaded")
        else:
            print("Warning: GEMINI_API_KEY not found in environment")
        
        print("AgriSage API server started successfully!")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

def get_context_from_db(location: str = None) -> Dict:
    """Get additional context from database based on location"""
    context = {}
    
    try:
        db_path = Path("data/agrisage.db")
        if not db_path.exists():
            return context
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if location:
            # Get recent weather for location
            cursor.execute("""
                SELECT precip_prob, max_temp, min_temp, NULL as soil_moisture 
                FROM reliable_weather w
                WHERE w.district LIKE ? 
                ORDER BY w.date DESC LIMIT 1
            """, (f"%{location}%",))
            
            result = cursor.fetchone()
            if result:
                context.update({
                    'precip_prob': result[0],
                    'max_temp': result[1], 
                    'min_temp': result[2],
                    'soil_moisture': result[3]
                })
        
        conn.close()
    except Exception as e:
        print(f"Error getting context from DB: {e}")
    
    return context

def get_query_intent(query: str) -> Dict[str, float]:
    """Classify query intent and extract keywords"""
    query_lower = query.lower()
    
    # Intent keywords with weights
    intent_patterns = {
        'irrigation': ['irrigat', 'water', 'watering', 'moisture', 'dry', 'wet'],
        'weather': ['weather', 'rain', 'temperature', 'forecast', 'climate'],
        'soil': ['soil', 'ph', 'nitrogen', 'phosphorus', 'potassium', 'nutrient'],
        'market': ['price', 'market', 'sell', 'buy', 'mandi', 'cost'],
        'fertilizer': ['fertiliz', 'nutrient', 'npk', 'urea', 'compost'],
        'pest': ['pest', 'insect', 'disease', 'spray', 'chemical']
    }
    
    intent_scores = {}
    for intent, keywords in intent_patterns.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            intent_scores[intent] = score / len(keywords)
    
    return intent_scores

def filter_by_metadata(documents: List[str], metadatas: List[Dict], query: str, location: str = None) -> tuple:
    """Filter documents by metadata relevance"""
    intent_scores = get_query_intent(query)
    
    if not intent_scores:
        return documents, metadatas, [1.0] * len(documents)
    
    # Get primary intent
    primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
    
    # Type mapping for filtering
    intent_to_type = {
        'irrigation': ['weather', 'soil'],
        'weather': ['weather'],
        'soil': ['soil'],
        'market': ['market', 'trade'],
        'fertilizer': ['soil'],
        'pest': ['weather', 'soil']
    }
    
    relevant_types = intent_to_type.get(primary_intent, [])
    
    filtered_docs = []
    filtered_metas = []
    relevance_scores = []
    
    for doc, meta in zip(documents, metadatas):
        base_score = 0.5  # Base relevance
        
        # Type relevance boost
        if meta.get('type') in relevant_types:
            base_score += 0.4
        
        # Location relevance boost
        if location and meta.get('district', '').lower() == location.lower():
            base_score += 0.3
        elif location and location.lower() in meta.get('district', '').lower():
            base_score += 0.2
        
        # Only keep documents with reasonable relevance
        if base_score >= 0.6:
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            relevance_scores.append(base_score)
    
    return filtered_docs, filtered_metas, relevance_scores

def retrieve_documents(query: str, k: int = 5, location: str = None) -> tuple:
    """Hybrid retrieval: vector similarity + metadata filtering + reranking"""
    try:
        # Get more candidates from vector search
        results = collection.query(
            query_texts=[query],
            n_results=min(k * 3, 15),  # Get 3x more candidates
            include=["documents", "metadatas", "distances"]
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] 
        distances = results['distances'][0]
        
        if not documents:
            return [], [], 0.0
        
        # Apply metadata filtering
        filtered_docs, filtered_metas, relevance_scores = filter_by_metadata(
            documents, metadatas, query, location
        )
        
        if not filtered_docs:
            # If no filtered results, return empty instead of irrelevant data
            logger.warning(f"No relevant data found for query: {query} (intent: {primary_intent})")
            return [], [], 0.0
        
        # Take top k filtered results
        final_docs = filtered_docs[:k]
        final_metas = filtered_metas[:k]
        final_scores = relevance_scores[:k]
        
        # Calculate average relevance score
        avg_retrieval_score = sum(final_scores) / len(final_scores) if final_scores else 0.0
        
        logger.info(f"Retrieved {len(final_docs)} filtered documents, avg score: {avg_retrieval_score:.3f}")
        return final_docs, final_metas, avg_retrieval_score
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return [], [], 0.0

def safety_gate_check(query: str, documents: List[str], metadatas: List[Dict], retrieval_score: float, llm_confidence: float) -> Dict:
    """Safety gate to prevent harmful advice without proper provenance"""
    
    # Check if query contains actionable keywords
    query_lower = query.lower()
    is_actionable_query = any(keyword in query_lower for keyword in ACTIONABLE_KEYWORDS)
    
    if not is_actionable_query:
        return {
            "safe": True,
            "actionable": False,
            "gate_reason": None
        }
    
    # Check provenance quality
    has_authoritative_source = any(
        meta.get('source', '') in AUTHORITATIVE_SOURCES 
        for meta in metadatas
    )
    
    # Check retrieval score threshold
    meets_score_threshold = retrieval_score >= MIN_PROVENANCE_SCORE
    
    # Combined confidence check
    combined_confidence = 0.6 * retrieval_score + 0.4 * llm_confidence
    meets_confidence_threshold = combined_confidence >= 0.5
    
    # Safety gate decision
    if has_authoritative_source and meets_score_threshold and meets_confidence_threshold:
        return {
            "safe": True,
            "actionable": True,
            "gate_reason": None
        }
    else:
        reasons = []
        if not has_authoritative_source:
            reasons.append("no authoritative sources")
        if not meets_score_threshold:
            reasons.append(f"low retrieval score ({retrieval_score:.2f} < {MIN_PROVENANCE_SCORE})")
        if not meets_confidence_threshold:
            reasons.append(f"low combined confidence ({combined_confidence:.2f} < 0.5)")
        
        return {
            "safe": False,
            "actionable": True,
            "gate_reason": "; ".join(reasons)
        }

def format_confidence_level(confidence: float) -> str:
    """Convert numeric confidence to human-readable level"""
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    else:
        return "Low"

def create_conservative_response(query: str, gate_reason: str) -> str:
    """Create conservative response when safety gate blocks actionable advice"""
    return f"""⚠️ **Insufficient authoritative data for actionable advice**

Your question appears to require specific agricultural guidance, but the available data doesn't meet our safety standards ({gate_reason}).

**Recommended actions:**
• Consult your local agricultural extension officer
• Visit the nearest Krishi Vigyan Kendra (KVK)
• Contact district agricultural department
• Speak with experienced farmers in your area

**Why we're being cautious:** Agricultural advice can significantly impact crop yields and farmer livelihoods. We only provide actionable recommendations when backed by authoritative government data sources."""

def log_llm_request(request_id: str, prompt: str, response: dict, status_code: int, latency: float, error: str = None):
    """Log LLM request for debugging and monitoring"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "model": "gemini-2.0-flash",
            "prompt_length": len(prompt),
            "status_code": status_code,
            "latency_ms": round(latency * 1000, 2),
            "success": status_code == 200,
            "error": error,
            "response_tokens": response.get("usageMetadata", {}).get("totalTokenCount", 0) if response else 0
        }
        
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        logger.error(f"Failed to log LLM request: {e}")

def call_gemini_llm(prompt: str) -> tuple:
    """Call Google Gemini LLM and return response with confidence"""
    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()
    
    try:
        if not gemini_api_key:
            logger.warning("Gemini API key not available")
            return None, 0.0
        
        # Gemini API endpoint - using gemini-2.0-flash (current available model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": f"You are AgriSage, an AI agricultural advisor for Indian farmers. Always end your response with a confidence score between 0.0 and 1.0.\n\n{prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        latency = (datetime.now() - start_time).total_seconds()
        
        if response.status_code == 200:
            result = response.json()
            log_llm_request(request_id, prompt, result, response.status_code, latency)
            
            if 'candidates' in result and len(result['candidates']) > 0:
                answer = result['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # Try to extract confidence from response
                llm_confidence = 0.7  # default
                if "confidence:" in answer.lower():
                    try:
                        conf_part = answer.lower().split("confidence:")[-1].strip()
                        conf_num = float(conf_part.split()[0])
                        if 0.0 <= conf_num <= 1.0:
                            llm_confidence = conf_num
                    except:
                        pass
                
                logger.info(f"LLM success [{request_id}]: {latency*1000:.0f}ms, confidence: {llm_confidence}")
                return answer, llm_confidence
            else:
                logger.warning(f"No candidates in Gemini response [{request_id}]")
                log_llm_request(request_id, prompt, result, response.status_code, latency, "No candidates")
                return None, 0.0
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error(f"Gemini API error [{request_id}]: {error_msg}")
            log_llm_request(request_id, prompt, {}, response.status_code, latency, error_msg)
            return None, 0.0
        
    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        logger.error(f"Error calling Gemini LLM [{request_id}]: {error_msg}")
        log_llm_request(request_id, prompt, {}, 0, latency, error_msg)
        return None, 0.0

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main RAG endpoint for agricultural questions"""
    
    # Safety check first - check for dangerous chemical/dosage queries
    dangerous_keywords = ['pesticide', 'insecticide', 'fungicide', 'herbicide', 'chemical', 'spray', 'dose', 'dosage', 'poison']
    if any(keyword in request.question.lower() for keyword in dangerous_keywords):
        return QueryResponse(
            answer="This question involves chemicals or dosages that require expert consultation. Please contact your local agricultural extension officer or Krishi Vigyan Kendra for safe recommendations.",
            confidence=1.0,
            provenance=[],
            escalate=True
        )
    
    try:
        # Retrieve relevant documents
        documents, metadatas, retrieval_score = retrieve_documents(request.question, location=request.location)
        
        if not documents:
            # Use fallback rules if no documents found
            context = get_context_from_db(request.location)
            fallback_result = get_fallback_response(request.question, context)
            
            return QueryResponse(
                answer=fallback_result["advice"],
                confidence=fallback_result["confidence"],
                provenance=[],
                escalate=fallback_result.get("escalate", False),
                fallback_used=True
            )
        
        # Build context for LLM
        context_text = "\n\n".join([
            f"Source: {meta['source']} (ID: {meta['row_id']})\nContent: {doc}"
            for doc, meta in zip(documents, metadatas)
        ])
        
        # Create prompt
        prompt = PROMPT_TEMPLATE.format(
            context=context_text,
            question=request.question,
            location=request.location or "Not specified"
        )
        
        # Call Gemini LLM
        llm_response, llm_confidence = call_gemini_llm(prompt)
        
        if not llm_response:
            # Fallback to rules engine
            context = get_context_from_db(request.location)
            fallback_result = get_fallback_response(request.question, context)
            
            return QueryResponse(
                answer=fallback_result["advice"],
                confidence=fallback_result["confidence"],
                provenance=[],
                escalate=fallback_result.get("escalate", False),
                fallback_used=True
            )
        
        # Calculate combined confidence
        combined_confidence = 0.6 * retrieval_score + 0.4 * llm_confidence
        
        # Apply safety gate
        safety_check = safety_gate_check(request.question, documents, metadatas, retrieval_score, llm_confidence)
        
        if not safety_check["safe"]:
            conservative_answer = create_conservative_response(request.question, safety_check["gate_reason"])
            
            return QueryResponse(
                answer=conservative_answer,
                confidence=combined_confidence,
                provenance=[{
                    "source": meta["source"],
                    "row_id": meta["row_id"],
                    "content": doc[:200] + "..." if len(doc) > 200 else doc
                } for meta, doc in zip(metadatas[:3], documents[:3])],
                escalate=True,
                actionable=safety_check["actionable"],
                safety_gate=safety_check["gate_reason"]
            )
        
        # Check if we should escalate for other reasons
        should_escalate = combined_confidence < 0.4 or "ESCALATE" in llm_response
        
        if should_escalate:
            context = get_context_from_db(request.location)
            fallback_result = get_fallback_response(request.question, context)
            
            return QueryResponse(
                answer=fallback_result["advice"],
                confidence=fallback_result["confidence"],
                provenance=[],
                escalate=True,
                fallback_used=True,
                actionable=safety_check["actionable"]
            )
        
        # Build enhanced provenance with URLs and dates
        provenance = []
        for meta, doc in zip(metadatas[:3], documents[:3]):
            prov_entry = {
                "source": meta["source"],
                "row_id": meta["row_id"],
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "date": meta.get("date", "Unknown"),
                "district": meta.get("district", "Unknown")
            }
            
            # Add source URLs where available
            source_urls = {
                "weather_forecast": "https://mausam.imd.gov.in",
                "soil_card": "https://soilhealth.dac.gov.in",
                "market_prices": "https://agmarknet.gov.in",
                "enam_trades": "https://enam.gov.in"
            }
            
            if meta["source"] in source_urls:
                prov_entry["url"] = source_urls[meta["source"]]
            
            provenance.append(prov_entry)
        
        # Enhanced response with safety metadata - deduplicate sources
        unique_sources = list(set([p['source'] for p in provenance]))
        enhanced_answer = f"{llm_response}\n\n**Sources:** {', '.join(unique_sources)}\n**Confidence:** {format_confidence_level(combined_confidence)}\n**Actionability:** {'Yes' if safety_check['actionable'] else 'No'}"
        
        return QueryResponse(
            answer=enhanced_answer,
            confidence=combined_confidence,
            provenance=provenance,
            escalate=False,
            actionable=safety_check["actionable"]
        )
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_records = 0
    try:
        db_path = Path("data/agrisage.db")
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
    except Exception as e:
        logger.error(f"Health check DB error: {e}")

    return {
        "status": "healthy",
        "chroma_connected": collection is not None,
        "model_loaded": sentence_model is not None,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "database_records": db_records,
        "vector_documents": collection.count() if collection else 0
    }

@app.post("/fallback")
async def fallback_endpoint(request: QueryRequest):
    """Direct access to fallback rules engine"""
    context = get_context_from_db(request.location)
    result = get_fallback_response(request.question, context)
    
    return QueryResponse(
        answer=result["advice"],
        confidence=result["confidence"],
        provenance=[],
        escalate=result.get("escalate", False),
        fallback_used=True
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
