# app/services/rag/prompt_builder.py
from typing import List, Dict, Any
import os

MAX_CHARS_CONTEXT = int(os.getenv("RAG_MAX_CHARS_CONTEXT", "3000"))
SYSTEM_INSTRUCTION = os.getenv("RAG_SYSTEM_INSTRUCTION",
    "You are a concise assistant answering questions about corporate filings and financial documents. " 
    "Use the provided document excerpts to answer; if not present, say you don't know. Provide source references."
)

def build_context_from_chunks(chunks: List[Dict[str, Any]], max_chars: int = MAX_CHARS_CONTEXT) -> str:
    """
    Assemble chunk_texts into a single context block, truncated to max_chars.
    Also attach a compact provenance line per chunk.
    """
    parts = []
    char_count = 0
    for c in chunks:
        snippet = c.get("chunk_text","").strip()
        if not snippet:
            continue
        prov = f"[doc:{c.get('document_id')} page:{c.get('page')} idx:{c.get('chunk_index')}]"
        block = f"{prov}\n{snippet}\n\n"
        if char_count + len(block) > max_chars:
            # try to include a trimmed version (end)
            remaining = max_chars - char_count
            if remaining <= 0:
                break
            block = block[:remaining]
            parts.append(block)
            break
        parts.append(block)
        char_count += len(block)
    return "\n".join(parts)

def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context = build_context_from_chunks(chunks)
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely. After the answer, list the provenance IDs used in the format: SOURCES: [doc:page:idx,...]"
    )
    return prompt
