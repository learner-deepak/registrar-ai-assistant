import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel


# Ensure Python path includes backend root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.rag_chain import generate_grounded_response

# Initialize FastAPI app
app = FastAPI(
    title="Registrar AI Assistant API",
    description="Backend API for querying university academic guidelines and registrar policies.",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing) for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend origin during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Calculate the exact path to your data/docs folder
DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

# BULLETPROOF FIX: Force create the directories if Render's server can't find them
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Mount that folder to a web URL endpoint called "/files"
app.mount("/files", StaticFiles(directory=str(DOCS_DIR)), name="files")

# Request Schema
class QueryRequest(BaseModel):
    query: str

# Response Schema 
class QueryResponse(BaseModel):
    answer: str
    citations: list[str]

@app.get("/")
def health_check():
    """Health check endpoint to verify server status."""
    return {
        "status": "online",
        "message": "Registrar AI Assistant API is operational."
    }

@app.post("/api/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """
    Accepts a user query, processes it through the RAG pipeline, 
    and returns a grounded answer with citations.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        result = generate_grounded_response(request.query)
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )