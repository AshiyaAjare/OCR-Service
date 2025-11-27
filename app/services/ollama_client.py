import httpx
from app.config import settings


async def call_ollama_mistral(prompt: str) -> str:
    """
    Calls local Ollama Mistral via /api/chat.
    Expects Ollama running at settings.OLLAMA_BASE_URL.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": settings.OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant that reads text extracted from PDFs "
                    "and returns concise, well-structured answers. "
                    "If user asks for JSON, respond with STRICT JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Ollama /api/chat returns:
    # { "message": { "role": "assistant", "content": "..." }, ... }
    return data.get("message", {}).get("content", "")
