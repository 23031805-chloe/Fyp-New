import streamlit as st
import query_util

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_ibm import WatsonxLLM

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 Document Q&A Chatbot")
st.markdown("Ask me anything about your documents!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    with st.spinner("🔄 Loading AI system..."):
        # Load vector database
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        # Setup LLM
        llm = WatsonxLLM(
            url="https://us-south.ml.cloud.ibm.com",
            apikey=os.getenv("WATSONX_APIKEY"),
            project_id=os.getenv("IBM_PROJECT_ID"),
            model_id="ibm/granite-3-8b-instruct",
            params={
                "temperature": 0.1,
                "max_new_tokens": 512,
                "repetition_penalty": 1.1
            }
        )
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create conversational chain
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
    
    st.success("✅ System ready!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"{i}. {source}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain({
                "question": prompt
            })
            
            answer = response["answer"]
            sources = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "content": doc.page_content
            }
            for doc in response["source_documents"]
        ]

            
            # Display answer
            st.markdown(answer)
            
            # Display sources
            with st.expander("📚 View Sources"):
                for i, src in enumerate(sources, 1):
                    st.write(f"### 📄 Source {i}")
                    st.write(f"**File:** {src['source']}")
                    st.write(f"**Page:** {src['page']}")
                    st.write(src["content"])
                    st.markdown("---")

            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    This chatbot uses:
    - **IBM Granite 3** LLM
    - **LangChain** framework
    - **Retrieval-Augmented Generation (RAG)**
    
    Ask questions about your documents and get accurate answers with sources!
    """)
    
    st.markdown("---")
    
    st.header("📊 Chat Statistics")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.qa_chain.memory.clear()
        st.rerun()
        

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.write("📄 Processing uploaded PDF...")

    file_path = f"./input/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    query_util.process_pdf(file_path)  # your existing PDF → chunks function
    st.success("PDF added successfully! Please reload the app.")
