from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_pdf import router as pdf_router
from app.api.routes_email import router as email_router
from app.api.routes_rag import router as rag_router
from app.api.routes_analysis import router as analysis_router

app = FastAPI(
    title="PDF Dual Extraction Service",
    version="0.1.0",
    description="Parallel PDF text & OCR extraction using FastAPI + Ollama Mistral",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(pdf_router)
app.include_router(email_router)
app.include_router(rag_router)
app.include_router(analysis_router)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
