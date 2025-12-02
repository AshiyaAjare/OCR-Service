from PIL.Image import logger
import httpx
from app.config import settings
import re
import json
from typing import Dict, Any, Optional


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
