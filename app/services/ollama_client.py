from PIL.Image import logger
import httpx
from app.config import settings
import re
import json
from typing import Dict, Any, Optional, List
import asyncio
import httpx
import os
import logging

logger = logging.getLogger(__name__)

# Default model/base url come from settings (you already import settings above)
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL")
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY", None)


async def call_ollama_mistral(prompt: str, temperature: float = 0.0, top_p: float = 0.1) -> str:
    """
    Calls local Ollama Mistral via /api/chat.
    Expects Ollama running at settings.OLLAMA_BASE_URL.
    
    Args:
        prompt: The user prompt to send
        temperature: Sampling temperature (0.0 for deterministic, default 0.0)
        top_p: Nucleus sampling parameter (lower = more deterministic, default 0.1)
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"

    # Strong JSON-only system prompt with few-shot examples
    system_prompt = """SYSTEM: You are a JSON-only extractor. Your task is to analyze PDF text and return EXACTLY ONE valid JSON object and nothing else.

    CRITICAL RULES:
    1. Output ONLY valid JSON - no markdown, no code fences, no explanations, no commentary
    2. Do not wrap JSON in ```json``` or any other formatting
    3. Do not add any text before or after the JSON object
    4. The JSON must be parseable by json.loads() directly

    EXAMPLE 1 - Good output:
    {"ticker": "STEELCAS", "company_name": "Steelcase Inc", "report_date": "2025-01-15", "period": "Q1 2025", "document_type": "results", "tables": [], "kv_pairs": []}

    EXAMPLE 2 - Good output:
    {"ticker": null, "company_name": "Patel Engineering Ltd", "report_date": "2025-11-26", "period": null, "document_type": "press_release", "tables": [], "kv_pairs": []}

    BAD - Do NOT output like this:
    - ```json {...} ```
    - "Here is the JSON: {...}"
    - Any text before or after the JSON object
    - Multiple JSON objects
    - Nested structures with roles/content_type/etc.

    Remember: Output ONLY the JSON object, nothing else."""

    payload = {
        "model": settings.OLLAMA_MODEL,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    async with httpx.AsyncClient(timeout=6000.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.ReadTimeout:
            logger.exception("Timeout calling Ollama/Mistral")
            raise  # re-raise and handle in the route
        except httpx.RequestError as e:
            logger.exception(f"Request error calling Ollama/Mistral: {e}")
            raise

    # Ollama /api/chat returns:
    # { "message": { "role": "assistant", "content": "..." }, ... }
    return data.get("message", {}).get("content", "")


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extracts a JSON object from text that may contain extra content.
    
    Strategies:
    1. Try direct json.loads()
    2. Find JSON object using regex (looks for {...})
    3. Try to find JSON array if object not found
    
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    
    if not text or not text.strip():
        return None
    
    # Strategy 1: Try direct parsing
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 2: Find JSON object using regex
    # Look for { ... } pattern, handling nested braces
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Strategy 3: Try to find JSON starting from first {
    first_brace = text.find('{')
    if first_brace != -1:
        # Try to find matching closing brace
        brace_count = 0
        for i in range(first_brace, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[first_brace:i+1]
                    try:
                        return json.loads(json_str)
                    except (json.JSONDecodeError, ValueError):
                        break
    
    return None

# -------------------------------------------------------------------------
# Synchronous helpers for embeddings + sync wrapper for Mistral (useful for
# code that is not async / quick scripts). Add below existing functions.
# -------------------------------------------------------------------------

def call_ollama_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 60.0,
) -> List[List[float]]:
    """
    Synchronous call to Ollama embedding endpoint.
    Returns list of embedding vectors (one per input text).

    Expected endpoints (tries /api/embed then /api/embeddings):
      POST {base_url}/api/embed   { "model": "...", "input": [...] }
      OR
      POST {base_url}/api/embeddings

    Handles a few common response shapes:
      - {"data": [{"embedding":[...]}, ...]}
      - {"embeddings": [[...], [...]]}
      - [{"embedding":[...]}, ...]
      - [[...], [...]]
    """
    model = model or DEFAULT_EMBED_MODEL
    base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    api_key = api_key or DEFAULT_API_KEY

    endpoint = f"{base_url}/api/embed"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "input": texts}

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            if resp.status_code != 200:
                # try alternate path
                alt = f"{base_url}/api/embeddings"
                logger.debug(f"{endpoint} returned {resp.status_code}, trying {alt}")
                resp = client.post(alt, json=payload, headers=headers)
            resp.raise_for_status()
            j = resp.json()
    except httpx.RequestError as e:
        logger.exception(f"Request error calling Ollama embeddings: {e}")
        raise

    embeddings: List[List[float]] = []

    # parse common shapes
    if isinstance(j, dict):
        if "data" in j and isinstance(j["data"], list):
            for item in j["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
                elif isinstance(item, list):
                    embeddings.append(item)
        elif "embeddings" in j and isinstance(j["embeddings"], list):
            embeddings.extend(j["embeddings"])
        elif "embedding" in j:
            # single embedding (or nested)
            e = j["embedding"]
            if isinstance(e, list) and isinstance(e[0], list):
                embeddings.extend(e)
            else:
                embeddings.append(e)
        else:
            # last resort: try to extract any lists of floats from values
            for v in j.values():
                if isinstance(v, list) and v and (isinstance(v[0], float) or isinstance(v[0], list)):
                    if isinstance(v[0], list):
                        embeddings.extend(v)  # multiple vectors
                    else:
                        embeddings.append(v)  # single vector
    elif isinstance(j, list):
        # Could already be list of vectors or list of dicts
        if j and isinstance(j[0], dict) and "embedding" in j[0]:
            for item in j:
                embeddings.append(item["embedding"])
        elif j and isinstance(j[0], list):
            embeddings.extend(j)
        else:
            # unexpected but try casting inner lists to vectors
            for item in j:
                if isinstance(item, list):
                    embeddings.append(item)

    if not embeddings:
        raise ValueError(f"Could not parse embedding response (shape unexpected): {j}")

    return embeddings


def call_ollama_mistral_sync(prompt: str, temperature: float = 0.0, top_p: float = 0.1, model: Optional[str] = None) -> str:
    """
    Synchronous wrapper around async call_ollama_mistral.
    Runs in asyncio.run(...) so only use from top-level sync code (not inside running loop).
    Returns the assistant content string.
    """
    model = model or getattr(settings, "OLLAMA_MODEL", None)

    # If your async call_ollama_mistral supported `model` param, you may pass it; otherwise
    # it uses settings.OLLAMA_MODEL. We keep signature similar to the async function.
    try:
        return asyncio.run(call_ollama_mistral(prompt, temperature=temperature, top_p=top_p))
    except RuntimeError as e:
        # This happens when there's already an event loop running (e.g., in uvicorn workers).
        # In that environment you should call the async function directly instead.
        logger.warning("asyncio.run failed (loop running). Falling back to httpx sync POST for /api/chat")
        # fallback synchronous POST
        base_url = getattr(settings, "OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        api_key = os.getenv("OLLAMA_API_KEY", DEFAULT_API_KEY)
        endpoint = f"{base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        system_prompt = ""  # minimal; the async function used a system_prompt earlier, but here we keep short
        payload = {
            "model": model or getattr(settings, "OLLAMA_MODEL", None),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "top_p": top_p},
            "stream": False,
        }
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "")

def call_ollama_chat_sync(
    prompt: str,
    model: str,
    system_prompt: str= "",
    temperature: float = 0.0,
    top_p: float = 0.9,
    timeout: float = 6000.0,
) -> str:
    """
    Generating natural langugage summaries for chat based answers
    """
    base_url = os.getenv("OLLAMA_BASE_URL")
    endpoint = f"{base_url}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt or "You are a helpful assistant.",
            },
            {
                "role" : "user",
                "content" : prompt,
            },
        ],
        "options": {
            "temperature" : temperature,
            "top_p": top_p,
        },
        "stream": False,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.RequestError as e:
        logger.exception(f"Request error calling Ollama chat: {e}")
        raise
    return data.get("message", {}).get("content", "").strip()