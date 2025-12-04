# app/services/rag/rag_pipeline.py
from typing import List, Dict, Any, Tuple, Optional
import logging
import json
import re
import os

from sqlalchemy.sql.functions import user

from app.services.ollama_client import (
    call_ollama_mistral_sync, 
    call_ollama_embeddings,
    call_ollama_chat_sync,
)
from app.services.rag.retreiver import retrieve_similar_chunks
from app.services.rag.prompt_builder import build_prompt

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "mistral:latest")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
SUMMARY_MODEL = os.getenv("RAG_SUMMARY_MODEL", "llama3.2:latest")


def parse_llm_response(llm_resp: str) -> Tuple[Any, List[str]]:
    """
    Parse an LLM response that looks like:

        {"ticker": null, ...}\n\nSOURCES: [doc:12 page:1 idx:0, doc:12 page:1 idx:1]

    Returns:
        answer_obj: parsed JSON object if possible, otherwise the raw string
        sources: list of source strings (e.g. "doc:12 page:1 idx:0")
    """
    # Separate the JSON-ish part from the "SOURCES:" trailer, if present
    parts = llm_resp.split("SOURCES:", 1)
    json_part = parts[0].strip()
    sources_raw = parts[1].strip() if len(parts) > 1 else ""

    # Try to parse the front as JSON; if it fails, return the raw string
    try:
        answer_obj: Any = json.loads(json_part)
    except json.JSONDecodeError:
        answer_obj = json_part

    # Extract sources inside square brackets: [ ... ]
    sources: List[str] = []
    if sources_raw:
        match = re.search(r"\[(.*)\]", sources_raw)
        if match:
            inner = match.group(1)
            # split on commas, strip whitespace, ignore empty entries
            sources = [s.strip() for s in inner.split(",") if s.strip()]

    return answer_obj, sources

def generate_answer_summary_with_llama(question: str, answer: Any) -> Optional[str]:
    """
    Use Ollama-hosted model to generate a concise natural-language 
    summary of the structured answer.
    """
    try:
        answer_json = json.dumps(answer, ensure_ascii=False, indent=2)
    except TypeError:
        answer_json = json.dumps({"value": str(answer)}, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a helpful assistant. Given a user question and a structured JSON answer,"
        "write a concise, natural-language summary of the answer.\n\n"
        "- Do not invent or infer any numbers, dates, percentages, amounts, or other details "
        "that are not explicitly present in the JSON.\n"
        "- If a specific quantity or amount is not given in the JSON, do NOT fabricate a value "
        "such as 0 or '$0'; instead, omit the quantity entirely.\n"
        "- Use plain, professional language.\n"
        "Refer to the entities exactly as they appear in the JSON.\n"
        "- If the answer is not clear, say you don't know.\n"
        "- If the answer is a date, format it as YYYY-MM-DD.\n"
        "- If the answer is a percentage, format it as a percentage (e.g. 10%).\n"
        "- If the answer is a boolean, format it as 'Yes' or 'No'.\n"
        "- If the answer is a list, format it as a comma-separated list.\n"
        "- If the answer is a dictionary, format it as a comma-separated list of key-value pairs.\n"
        "- OUTPUT FORMAT CONSTRAINTS:\n"
        "  * Respond with plain English sentences only.\n"
        "  * Do NOT use bullet points, dashes, numbered lists, or any markdown formatting.\n"
        "  * Do NOT include blank lines or line breaks inside the summary.\n"
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Structured answer JSON: \n\n"
        f"{answer_json}"
    )

    try:
        summary = call_ollama_chat_sync(
            prompt=user_prompt,
            model=SUMMARY_MODEL,
            system_prompt=system_prompt,
            temperature=0.0,
            top_p=0.9,
        )
        summary = (summary or "").strip()
        return summary or None
    except Exception as e:
        logger.exception(f"Error generating summary with {SUMMARY_MODEL}: {e}")
        return None

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

    # 4) call LLM
    llm_resp = call_ollama_mistral_sync(prompt, temperature=0.0, top_p=0.1, model=LLM_MODEL)

    # Parse LLM response into structured JSON + sources, if possible
    answer_obj, sources = parse_llm_response(llm_resp)

    # optional: extract structured provenance
    provenance = [
        {
            "document_id": c["document_id"], 
            "page": c["page"], 
            "chunk_index": c["chunk_index"]
        } 
        for c in chunks
    ]

    summary = generate_answer_summary_with_llama(question, answer_obj)

    return {
        "answer": answer_obj,
        "summary": summary,
        "sources": sources,
        "provenance": provenance,
        "used_chunks_count": len(chunks),
    }
