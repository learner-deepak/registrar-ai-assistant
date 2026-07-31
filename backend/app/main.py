import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Rate Limiting Imports ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Ensure Python path includes backend root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.rag_chain import generate_grounded_response

# 1. Initialize Limiter (tracks users by client IP address)
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Registrar AI Assistant API",
    description="Backend API for querying university academic guidelines and registrar policies.",
    version="1.0.0"
)

# 2. Attach Limiter to app state & register error handler for 429 responses
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. Define allowed origins (Restricts access to only your frontend domains)
origins = [
    "https://ro-assistant.vercel.app/",  # Replace with your exact Vercel frontend URL
    "http://localhost:3000",                      # React / Next.js local dev
    "http://localhost:5173",                      # Vite local dev
    "http://127.0.0.1:5500",                     # Live Server local dev
]

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Calculate the exact path to your data/docs folder
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "docs"

# Force create the directories if server can't find them
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

@app.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")  # Limit to 5 requests per minute per IP
def handle_query(request: Request, body: QueryRequest):
    """
    Accepts a user query, processes it through the RAG pipeline, 
    and returns a grounded answer with citations. Rate limited to 5 req/min.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        result = generate_grounded_response(body.query)
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )