from typing import List
import pdfplumber


def extract_text_from_pdf_pages(file_path: str) -> List[str]:
    """
    Extracts text for each page using a PDF text parser.
    Returns a list where index 0 is page 1, etc.
    """
    texts: List[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            texts.append(text)
    return texts
