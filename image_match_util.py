import io
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_NAME = "openai/clip-vit-base-patch32"

_clip_model = CLIPModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
_clip_processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
_clip_model.eval()


def _to_rgb_pil(img_or_bytes) -> Image.Image:
    if isinstance(img_or_bytes, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(img_or_bytes))).convert("RGB")
    if isinstance(img_or_bytes, Image.Image):
        return img_or_bytes.convert("RGB")
    raise TypeError("Expected PIL.Image or image bytes")


@torch.no_grad()
def clip_image_embedding(img_or_bytes) -> np.ndarray:
    img = _to_rgb_pil(img_or_bytes)
    inputs = _clip_processor(images=img, return_tensors="pt").to(_DEVICE)
    feats = _clip_model.get_image_features(**inputs)  # [1,512]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].detach().cpu().numpy().astype(np.float32)


def save_index(meta_path: str, npy_path: str, records: List[Dict], emb_matrix: np.ndarray) -> None:
    import os, json
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2)

    np.save(npy_path, emb_matrix)


def load_index(meta_path: str, npy_path: str) -> Tuple[List[Dict], Optional[np.ndarray]]:
    import os, json
    if not (os.path.exists(meta_path) and os.path.exists(npy_path)):
        return [], None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mat = np.load(npy_path)
    return meta.get("records", []), mat


def best_match(uploaded_img_bytes: bytes, records: List[Dict], emb_matrix: np.ndarray) -> Optional[Dict]:
    if emb_matrix is None or len(records) == 0:
        return None

    q = clip_image_embedding(uploaded_img_bytes)  # (512,)
    scores = emb_matrix @ q  # cosine since normalized
    idx = int(np.argmax(scores))
    return {"path": records[idx]["path"], "score": float(scores[idx])}
