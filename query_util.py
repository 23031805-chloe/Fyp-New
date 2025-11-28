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
def process_pdf(
    pdf_path,
    persist_directory="./chroma_db",
    embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Load a PDF, split it into chunks, embed, and store in Chroma DB
    """

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name
    )

    # Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectorstore.persist()

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
                "max_new_tokens": 512,
                "repetition_penalty": 1.1
            }
        )
    else:
        raise ValueError("Only 'ibm/granite-3-8b-instruct' is supported.")

    # Prompt template
    prompt_template = """
    Use the following context to answer the question.
    If the answer cannot be found in the context, say:
    "I cannot find this information in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
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
