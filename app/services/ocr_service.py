from typing import List
from PIL import Image
import pytesseract


def run_ocr_on_images(images: List[Image.Image], lang: str = "eng") -> List[str]:
    """
    Runs Tesseract OCR on each page image.
    Returns a list of recognized text per page.
    """
    texts: List[str] = []
    for img in images:
        text = pytesseract.image_to_string(img, lang=lang) or ""
        texts.append(text)
    return texts
