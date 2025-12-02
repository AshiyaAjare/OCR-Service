# app/services/rag/rag_pipeline.py
from typing import List, Dict, Any
import logging
from app.services.ollama_client import call_ollama_mistral_sync, call_ollama_embeddings
from app.services.rag.retreiver import retrieve_similar_chunks
from app.services.rag.prompt_builder import build_prompt
import os

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "mistral:latest")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

def answer_question_with_rag(question: str, top_k: int = TOP_K, conn=None) -> Dict[str, Any]:
    """
    Steps:
     1) embed user question
     2) retrieve k chunks from pgvector
     3) build prompt
     4) call LLM to produce answer
     5) return answer + provenance
    """
    # 1) embed question (reuses your Ollama embedding function)
    q_embed = call_ollama_embeddings([question], model=EMBED_MODEL)[0]

    # 2) retrieve chunks
    chunks = retrieve_similar_chunks(q_embed, top_k=top_k, conn=conn)

    # 3) build prompt
    prompt = build_prompt(question, chunks)

    # 4) call LLM (you have call_ollama_mistral in project)
    # The function signature might be different in your ollama_client - adapt accordingly
    llm_resp = call_ollama_mistral_sync(prompt, temperature=0.0, top_p=0.1, model=LLM_MODEL)

    # optional: extract structured provenance
    provenance = [{"document_id": c["document_id"], "page": c["page"], "chunk_index": c["chunk_index"]} for c in chunks]

    return {
        "answer": llm_resp,
        "provenance": provenance,
        "used_chunks_count": len(chunks),
    }
