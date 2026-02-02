import os
import re
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# ✅ Stronger retrieval embeddings (must match chatbot.py)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

REFERENCE_PATTERNS = [
    r"\bretrieved\b",
    r"\bdoi\b",
    r"https?://",
    r"\b(19|20)\d{2}\b",
    r"\bet\s+al\.?\b",
]

def clean_text(text: str) -> str:
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # Fix hyphenated line-break words
    text = re.sub(r"\s+", " ", text).strip()       # Normalize whitespace
    return text

def _looks_like_references(text: str) -> bool:
    t = (text or "").lower()
    hits = sum(1 for p in REFERENCE_PATTERNS if re.search(p, t))
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    many_short_lines = (
        len(lines) >= 6 and sum(1 for ln in lines if len(ln) <= 60) / max(1, len(lines)) > 0.7
    )
    return hits >= 3 or (many_short_lines and hits >= 2)

class FilteredRetriever(BaseRetriever):
    """
    Retriever that filters out pages that look like reference lists.
    Falls back to 'definition' retrieval if everything is filtered.
    """
    def __init__(self, base_retriever: BaseRetriever, fallback_retriever: Optional[BaseRetriever] = None):
        super().__init__()
        self.base_retriever = base_retriever
        self.fallback_retriever = fallback_retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        filtered = [d for d in docs if not _looks_like_references(d.page_content)]
        if self.fallback_retriever and not filtered:
            docs2 = self.fallback_retriever.get_relevant_documents(query + " definition")
            filtered2 = [d for d in docs2 if not _looks_like_references(d.page_content)]
            if filtered2:
                return filtered2
        return filtered or docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        docs = await self.base_retriever.aget_relevant_documents(query)
        filtered = [d for d in docs if not _looks_like_references(d.page_content)]
        if self.fallback_retriever and not filtered:
            docs2 = await self.fallback_retriever.aget_relevant_documents(query + " definition")
            filtered2 = [d for d in docs2 if not _looks_like_references(d.page_content)]
            if filtered2:
                return filtered2
        return filtered or docs

def _load_documents(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
        docs = loader.load()
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    for d in docs:
        d.page_content = clean_text(d.page_content)
        d.metadata["source"] = path

    return docs

def _split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)

def process_document(path: str, persist_directory: str = "./chroma_db"):
    """
    Index a document into an existing Chroma DB (or create one if missing).
    IMPORTANT: supports multiple uploads (append).
    """
    os.makedirs(persist_directory, exist_ok=True)

    docs = _load_documents(path)
    chunks = _split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Load or create, then add
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectordb.add_documents(chunks)

    # Persist for compatibility across versions
    try:
        vectordb.persist()
    except Exception:
        pass

    return vectordb

def process_pdf(pdf_path: str, persist_directory: str = "./chroma_db", embeddings_model_name: str = EMBEDDING_MODEL):
    # Backwards compatibility
    return process_document(pdf_path, persist_directory=persist_directory)

def setup_qa_chain(local_vector_store_path="./chroma_db", use_filtered_retriever: bool = False):
    """
    Optional QA chain (not used by chatbot.py, but kept for testing).
    - use_filtered_retriever=True avoids reference-list pages.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=local_vector_store_path, embedding_function=embeddings)

    base_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5}
    )

    retriever = FilteredRetriever(base_retriever, base_retriever) if use_filtered_retriever else base_retriever

    prompt_template = """
You are an expert assistant.

Answer the question based ONLY on the context below.
If the answer is not in the context, say:
"I cannot find this information in the document."

Context:
{context}

Question:
{question}

Instructions:
1. Be direct and factual.
2. Include specific dates, names, numbers, or steps if they appear.
3. Prefer 2–4 bullet points when possible.

Answer:
"""
    PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    llm = WatsonxLLM(
        url="https://us-south.ml.cloud.ibm.com",
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("IBM_PROJECT_ID"),
        model_id="ibm/granite-3-8b-instruct",
        params={
            "temperature": 0.1,
            "max_new_tokens": 500,
            "min_new_tokens": 1
        },
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return qa_chain
