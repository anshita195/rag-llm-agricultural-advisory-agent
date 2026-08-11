#!/usr/bin/env python3
"""FastAPI server for AgriSage RAG system."""

from pathlib import Path
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from rag.pipeline import ask, get_health, initialize, get_context_from_db
from rules_engine.fallback import get_fallback_response

app = FastAPI(title="AgriSage API", description="Agricultural Advisory RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """Initialize pipeline on startup."""
    initialize()


def _to_response(result: Dict) -> QueryResponse:
    return QueryResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        provenance=result.get("provenance", []),
        escalate=result.get("escalate", False),
        fallback_used=result.get("fallback_used", False),
        actionable=result.get("actionable", False),
        safety_gate=result.get("safety_gate"),
    )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main RAG endpoint for agricultural questions."""
    try:
        result = ask(
            question=request.question,
            location=request.location,
            user_id=request.user_id,
        )
        return _to_response(result)
    except Exception as exc:
        print(f"Error processing question: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return get_health()


@app.post("/fallback", response_model=QueryResponse)
async def fallback_endpoint(request: QueryRequest):
    """Direct access to fallback rules engine."""
    context = get_context_from_db(request.location)
    result = get_fallback_response(request.question, context)
    return QueryResponse(
        answer=result["advice"],
        confidence=result["confidence"],
        provenance=[],
        escalate=result.get("escalate", False),
        fallback_used=True,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
