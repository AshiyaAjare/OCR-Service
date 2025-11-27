import os


class Settings:
    def __init__(self) -> None:
        # Ollama base URL & model name
        self.OLLAMA_BASE_URL: str = os.getenv(
            "OLLAMA_BASE_URL", "http://192.168.2.23:11435"
        )
        self.OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

        # Where to temporarily store uploaded PDFs
        self.UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/pdf_extractor")


settings = Settings()
