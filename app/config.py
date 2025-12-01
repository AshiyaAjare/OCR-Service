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

        self.gmail_imap_host: str = os.getenv("GMAIL_IMAP_HOST")
        if not self.gmail_imap_host:
            raise ValueError("GMAIL_IMAP_HOST must be set in .env file")
        
        self.gmail_email: str = os.getenv("GMAIL_EMAIL")
        if not self.gmail_email:
            raise ValueError("GMAIL_EMAIL must be set in .env file")
        
        self.gmail_app_password: str = os.getenv("GMAIL_APP_PASSWORD")

        # Database configuration
        self.DATABASE_NAME: str = os.getenv("DATABASE_NAME")
        self.DATABASE_USER: str = os.getenv("DATABASE_USER")
        self.DATABASE_PASSWORD: str = os.getenv("DATABASE_PASSWORD")
        self.DATABASE_HOST: str = os.getenv("DATABASE_HOST")
        self.DATABASE_PORT: str = os.getenv("DATABASE_PORT")
        
        # Table prefix to avoid conflicts with Django project's tables
        self.DATABASE_TABLE_PREFIX: str = os.getenv("DATABASE_TABLE_PREFIX", "ocr_")
        
        # Build database URL
        self.DATABASE_URL: str = (
            f"postgresql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )


settings = Settings()
