from fastapi import FastAPI
from app.api.routes_pdf import router as pdf_router

app = FastAPI(
    title="PDF Dual Extraction Service",
    version="0.1.0",
    description="Parallel PDF text & OCR extraction using FastAPI + Ollama Mistral",
)

app.include_router(pdf_router)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
