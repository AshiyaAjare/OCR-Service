from typing import List
from pdf2image import convert_from_path
from PIL import Image


def render_pdf_to_images(file_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Renders each page of the PDF as a PIL Image.
    Page 1 -> index 0, etc.
    """
    images: List[Image.Image] = convert_from_path(file_path, dpi=dpi)
    return images
