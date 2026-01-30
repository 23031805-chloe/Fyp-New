from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os


# --------------------------------------------------
# PDF → Chroma ingestion
# --------------------------------------------------
def process_document(
    file_path,
    persist_directory="./chroma_db",
    embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Load a document (PDF, DOCX, or TXT), split it into chunks, embed, and store in Chroma DB
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
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name
    )

    # Load existing vectorstore or create new
    if os.path.exists(persist_directory):
        # Add to existing database
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)
    else:
        # Create new database
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )

    return vectorstore


# --------------------------------------------------
# RAG setup (LLM + Retriever)
# --------------------------------------------------
def setup_qa_chain(
    local_vector_store_path=None,
    vector_object=None,
    use_local_path=True,
    model_id="ibm/granite-3-8b-instruct",
    embbedings_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Set up a RetrievalQA RAG chain using Chroma + IBM Granite
    """

    # Configure IBM Granite model
    if model_id == "ibm/granite-3-8b-instruct":
        llm = WatsonxLLM(
            url="https://us-south.ml.cloud.ibm.com",
            apikey=os.environ.get("WATSONX_APIKEY"),
            project_id=os.environ.get("IBM_PROJECT_ID"),
            model_id=model_id,
            params={
                "temperature": 0.1,
                "max_new_tokens": 350,
                "repetition_penalty": 1.1
            }
        )
    else:
        raise ValueError("Only 'ibm/granite-3-8b-instruct' is supported.")

    # Prompt template
    
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
    - You may combine or restructure information from the context.
    - Only use the context for factual grounding. Do NOT hallucinate new facts.
    - If the information is NOT in the context, reply:
      "I cannot find this information in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Your answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


    # Load vector store
    if use_local_path:
        if not local_vector_store_path:
            raise ValueError("`local_vector_store_path` is required.")

        embeddings = HuggingFaceEmbeddings(
            model_name=embbedings_model_name
        )

        retriever_source = Chroma(
            persist_directory=local_vector_store_path,
            embedding_function=embeddings
        )
    else:
        if vector_object is None:
            raise ValueError("`vector_object` must be provided.")
        retriever_source = vector_object

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_source.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain


# --------------------------------------------------
# Question helper
# --------------------------------------------------
def ask_question(qa_chain, question):
    """
    Ask a question and return answer + sources
    """
    result = qa_chain({"query": question})

    return {
        "answer": result["result"],
        "sources": [
            doc.metadata.get("source", "Unknown")
            for doc in result["source_documents"]
        ],
        "confidence": len(result["source_documents"])
    }
