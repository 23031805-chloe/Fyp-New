# ocr_util.py
import re
from typing import List

def _clean_lines(lines: List[str]) -> str:
    text = "\n".join([l.strip() for l in lines if l and l.strip()])
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def extract_text_from_image(pil_img) -> str:
    """
    OCR text from an image (best effort).
    Uses EasyOCR if installed. If not installed, returns "".
    """
    try:
        import numpy as np
        import easyocr
    except Exception:
        return ""

    try:
        reader = easyocr.Reader(["en"], gpu=False)
        arr = np.array(pil_img)
        results = reader.readtext(arr, detail=0)  # returns list[str]
        return _clean_lines(results)
    except Exception:
        return ""
