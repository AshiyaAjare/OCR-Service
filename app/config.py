import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    def __init__(self) -> None:
        # Ollama base URL & model name
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL")
        if not self.OLLAMA_BASE_URL:
            raise ValueError("OLLAMA_BASE_URL must be set in .env file")
            
        self.OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

        # Where to temporarily store uploaded PDFs
        self.UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/pdf_extractor")


settings = Settings()
