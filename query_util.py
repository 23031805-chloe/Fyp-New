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


REFERENCE_PATTERNS = [
    r"\bretrieved\b",
    r"\bdoi\b",
    r"https?://",
    r"\b(19|20)\d{2}\b",
    r"\bet\s+al\.?\b",
]


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
    Falls back to "definition" retrieval if everything is filtered.
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


# --------------------------------------------------
# Document → Chroma ingestion (PDF / DOCX / TXT)
# --------------------------------------------------
def process_document(
    file_path,
    persist_directory="./chroma_db",
    embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Load a document (PDF, DOCX, or TXT), split it into chunks, embed, and store in Chroma DB.
    If the DB already exists, add new chunks to it.
    """
    from pathlib import Path

    file_extension = Path(file_path).suffix.lower()

    # Load document based on file type
    if file_extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif file_extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(file_path)
        documents = loader.load()

    elif file_extension == ".txt":
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Split into chunks (keep Wahyu separators for better chunking)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Load existing vectorstore or create new
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

    vectorstore.persist()
    return vectorstore


def process_pdf(
    pdf_path,
    persist_directory="./chroma_db",
    embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Backward compatible wrapper (some old code calls process_pdf).
    """
    return process_document(
        file_path=pdf_path,
        persist_directory=persist_directory,
        embeddings_model_name=embeddings_model_name,
    )


# --------------------------------------------------
# QA Chain setup
# --------------------------------------------------
def setup_qa_chain(local_vector_store_path="./chroma_db", use_filtered_retriever=False):
    """
    Create a RetrievalQA chain.
    - use_filtered_retriever=False by default to keep Kaixin/login behavior.
    - set True if you want to avoid "reference pages" (Wahyu).
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=local_vector_store_path, embedding_function=embeddings)

    base_retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Optional filtered retriever (Wahyu)
    if use_filtered_retriever:
        retriever = FilteredRetriever(base_retriever=base_retriever, fallback_retriever=base_retriever)
    else:
        retriever = base_retriever

    # Prompt merged (KEEP BOTH sets of rules)
    prompt_template = """
You are an intelligent AI assistant answering questions using the document context provided.

IMPORTANT RULES:
- Always rewrite and reorganize the information naturally — do NOT copy the document formatting.
- Adjust your answer style based on USER INTENT:
    • If the user says "summarise", give a concise summary.
    • If the user says "explain", "elaborate", "long answer", give a detailed explanation.
    • If the user says "list", "objectives", "importance", "benefits", "factors", produce bullet points.
    • If the user says "give examples", include examples.
- Write clean, easy-to-read answers like ChatGPT.
- Only use the context for factual grounding. Do NOT hallucinate new facts.
- If the information is NOT in the context, reply:
  "I cannot find this information in the provided documents."
- If the context contains a definition, explain it clearly.
- If the context is just a Table of Contents, say:
  "I found this in the Table of Contents, but I need to search the actual chapter."
- Prefer 2-3 concise bullet points unless the user explicitly asks for a long explanation.

Context:
{context}

Question:
{question}

Your answer:
"""

    PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    llm = WatsonxLLM(
        url="https://us-south.ml.cloud.ibm.com",
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("IBM_PROJECT_ID"),
        model_id="ibm/granite-3-8b-instruct",
        params={
            "temperature": 0.1,
            "max_new_tokens": 350,
            "repetition_penalty": 1.1,
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


def ask_question(qa_chain, question):
    """
    Run the QA chain and return answer + UNIQUE sources.
    """
    result = qa_chain({"query": question})

    answer = result.get("result", "")
    source_docs = result.get("source_documents", []) or []

    unique = {}
    for doc in source_docs:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        content = (doc.page_content or "").strip()
        key = (src, str(page))

        if key not in unique or len(content) > len(unique[key]["content"]):
            unique[key] = {"source": src, "page": page, "content": content}

    formatted_sources = list(unique.values())[:3]
    return {"answer": answer, "sources": formatted_sources, "confidence": len(formatted_sources)}
