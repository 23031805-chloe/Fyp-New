import os
import re
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever


REFERENCE_PATTERNS = [r"\bretrieved\b", r"\bdoi\b", r"https?://", r"\b(19|20)\d{2}\b", r"\bet\s+al\.?\b"]

def _looks_like_references(text: str) -> bool:
    t = (text or "").lower()
    hits = sum(1 for p in REFERENCE_PATTERNS if re.search(p, t))
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    many_short_lines = len(lines) >= 6 and sum(1 for ln in lines if len(ln) <= 60) / max(1, len(lines)) > 0.7
    return hits >= 3 or (many_short_lines and hits >= 2)

class FilteredRetriever(BaseRetriever):
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


def process_pdf(
    pdf_path,
    persist_directory="./chroma_db",
    embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """Load PDF, chunk text, embed, store in Chroma."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    vectordb.persist()
    return vectordb


def setup_qa_chain(local_vector_store_path="./chroma_db"):
    """
    Standard QA chain that DOES NOT filter out pages.
    """
    # 1. Load the database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=local_vector_store_path, embedding_function=embeddings)

    # 2. Use a standard retriever (Removed "FilteredRetriever")
    # We increase k=5 to get more context
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # 3. Use a prompt that forces the AI to look for definitions
    prompt_template = """
    You are a helpful research assistant. Use the provided context to answer the question.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Instructions:
    - If the context contains a definition, explain it clearly.
    - If the context is just a Table of Contents, say "I found this in the Table of Contents, but I need to search the actual chapter."
    - Answer in 2-3 concise bullet points.
    
    Answer:
    """

    PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # 4. Configure the LLM
    llm = WatsonxLLM(
        url="https://us-south.ml.cloud.ibm.com",
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("IBM_PROJECT_ID"),
        model_id="ibm/granite-3-8b-instruct",
        params={
            "temperature": 0.1,
            "max_new_tokens": 300,
            "repetition_penalty": 1.1
        },
    )

    # 5. Build the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return qa_chain


def ask_question(qa_chain, question):
    """Run the QA chain and return answer + UNIQUE sources."""
    result = qa_chain({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    unique = {}
    for doc in source_docs:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        key = (src, str(page))

        if key not in unique or len(content) > len(unique[key]["content"]):
            unique[key] = {"source": src, "page": page, "content": content}

    formatted_sources = list(unique.values())[:3]
    return {"answer": answer, "sources": formatted_sources, "confidence": len(formatted_sources)}
