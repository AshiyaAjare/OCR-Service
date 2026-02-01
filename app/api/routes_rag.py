# app/api/routes_rag.py
from typing import Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.services.rag.rag_pipeline import answer_question_with_rag

router = APIRouter(prefix="/rag", tags=["RAG"])

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: Any  # now allows the parsed JSON object
    summary: Optional[str] = None
    sources: List[str]  # new field for the parsed SOURCES list
    provenance: list
    used_chunks_count: int

@router.post("/qa", response_model=QueryResponse)
def rag_qa(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    result = answer_question_with_rag(req.question, top_k=req.top_k)
    return result
