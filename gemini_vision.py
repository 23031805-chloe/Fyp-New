# gemini_vision.py
import os
import time
import json
import base64
import requests
from typing import Optional

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def _post_gemini(payload: dict, retries: int = 4) -> Optional[dict]:
    if not GEMINI_API_KEY:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    backoff = 1.5
    for _ in range(retries):
        try:
            r = requests.post(url, headers=headers, params=params, data=json.dumps(payload), timeout=60)
            if r.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(backoff)
            backoff *= 2
    return None


def _extract_text(data: dict) -> Optional[str]:
    if not data:
        return None
    cand = data.get("candidates", [])
    if not cand:
        return None
    parts = cand[0].get("content", {}).get("parts", [])
    if not parts:
        return None
    return parts[0].get("text")


def answer_from_context(question: str, context: str) -> Optional[str]:
    prompt = f"""You are a document QA assistant.

RULES:
- Answer using ONLY the provided context.
- If the answer is not in the context, say: "Not found in the selected document."
- Keep it clear and direct.

CONTEXT:
{context}

QUESTION:
{question}
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }
    data = _post_gemini(payload)
    out = _extract_text(data)
    return out.strip() if out else None


def extract_text_from_image(image_bytes: bytes, mime_type: str = "image/png") -> Optional[str]:
    """
    Multimodal OCR-like extraction.
    IMPORTANT: pass correct mime_type (image/jpeg for jpg).
    """
    if not GEMINI_API_KEY or not image_bytes:
        return None

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "Extract the visible text from this image as accurately as possible. "
        "Focus on names, titles, headings, and captions. "
        "Return ONLY the extracted text (no explanation)."
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64}}
            ]
        }],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }

    data = _post_gemini(payload, retries=3)
    out = _extract_text(data)
    return out.strip() if out else None





