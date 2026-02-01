import os
import json
import hashlib
from typing import List, Dict

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# If your docling file is docling_util.py, keep this:
import docling_util


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MD_DIR = os.path.join(BASE_DIR, "output_md")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
MANIFEST_PATH = os.path.join(CHROMA_DIR, "manifest.json")

os.makedirs(OUTPUT_MD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def _normalize_path(p: str) -> str:
    return p.replace("\\", "/")

def _load_manifest() -> Dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_manifest(m: Dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

def _safe_collection_name(md_path: str) -> str:
    h = hashlib.sha1(_normalize_path(md_path).encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(md_path))[0]
    return f"{base[:30]}_{h}"

def read_md(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

def process_pdf(pdf_path: str) -> str:
    """
    Convert PDF -> markdown using docling_util.process_pdf
    """
    md_path = docling_util.process_pdf(pdf_path)
    return _normalize_path(md_path)

def ensure_indexed(md_path: str) -> None:
    md_path = _normalize_path(md_path)
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"MD not found: {md_path}")

    m = _load_manifest()
    mtime = os.path.getmtime(md_path)
    entry = m.get(md_path)
    collection = _safe_collection_name(md_path)

    if entry and entry.get("mtime") == mtime:
        return

    text = read_md(md_path)
    chunks = SPLITTER.split_text(text)

    vectordb = Chroma(
        collection_name=collection,
        persist_directory=CHROMA_DIR,
        embedding_function=EMBEDDINGS,
    )

    # clear old data
    try:
        existing = vectordb.get()
        if existing and existing.get("ids"):
            vectordb.delete(ids=existing["ids"])
    except Exception:
        pass

    # add new docs
    metadatas = [{"source": md_path, "chunk_id": i} for i in range(len(chunks))]
    vectordb.add_texts(chunks, metadatas=metadatas)
    vectordb.persist()

    m[md_path] = {"mtime": mtime, "chunks": len(chunks), "collection": collection}
    _save_manifest(m)

def retrieve_from_source(query: str, md_path: str, k: int = 8):
    md_path = _normalize_path(md_path)
    m = _load_manifest()
    entry = m.get(md_path)

    if not entry:
        ensure_indexed(md_path)
        m = _load_manifest()
        entry = m.get(md_path)

    collection = entry["collection"]

    vectordb = Chroma(
        collection_name=collection,
        persist_directory=CHROMA_DIR,
        embedding_function=EMBEDDINGS,
    )
    return vectordb.similarity_search(query, k=k)


