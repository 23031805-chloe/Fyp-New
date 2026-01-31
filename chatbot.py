import os
import re
import hashlib

import streamlit as st
import query_util

from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ibm import WatsonxLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ----------------------------
# IMPROVED: Generate context-aware follow-up questions
# ----------------------------
def generate_followup_questions(answer, original_question, conversation_history=None):
    """
    Generate smart, context-aware follow-up questions that are unique to each answer.
    
    Args:
        answer: The AI's response
        original_question: The user's question
        conversation_history: Previous messages to avoid repetition
    """
    followups = []
    answer_lower = answer.lower()
    question_lower = original_question.lower()
    
    # Get previously asked follow-ups to avoid repetition
    asked_questions = set()
    if conversation_history:
        for msg in conversation_history:
            if msg.get("role") == "assistant" and "followup_questions" in msg:
                asked_questions.update(msg["followup_questions"])
    
    # Extract MEANINGFUL entities (not generic terms)
    entities = extract_meaningful_entities(answer, original_question)
    
    # CRITICAL: If answer says "not found" or "no information", use different strategy
    if any(phrase in answer_lower for phrase in [
        "does not provide", "cannot find", "no information", 
        "not mentioned", "not specified", "would be required"
    ]):
        # Answer didn't have info - suggest related questions
        main_topic = extract_topic_from_question(original_question)
        if main_topic:
            fallback_questions = [
                f"What information about {main_topic} is available in the document?",
                f"Can you tell me about related topics in the document?",
                "What are the main topics covered in this document?",
            ]
        else:
            fallback_questions = [
                "What are the main topics covered in this document?",
                "Can you provide a summary of the document?",
                "What key information is available in the document?",
            ]
        
        for q in fallback_questions:
            if q not in asked_questions and len(followups) < 3:
                followups.append(q)
        
        return followups[:3]
    
    # Strategy 1: Use MEANINGFUL entities from answer
    if entities:
        entity = entities[0]  # Use the most prominent meaningful entity
        
        # Generate questions that make sense with the entity
        entity_questions = [
            f"Can you explain more about {entity}?",
            f"What are the key aspects of {entity}?",
            f"How is {entity} implemented or used?",
            f"What are examples of {entity}?",
            f"What challenges are associated with {entity}?",
        ]
        
        for q in entity_questions:
            if q not in asked_questions and len(followups) < 3:
                followups.append(q)
    
    # Strategy 2: Content-based intelligent questions (improved)
    if len(followups) < 3:
        if any(word in answer_lower for word in ['phase', 'week', 'stage', 'step', 'process']):
            candidates = [
                "What happens in the next phase?",
                "What are the deliverables for this phase?",
                "What skills are needed for this phase?",
            ]
        elif any(word in answer_lower for word in ['technology', 'tool', 'framework', 'system']):
            candidates = [
                "What are the advantages of using this technology?",
                "How do you get started with this?",
                "What are common use cases?",
            ]
        elif any(word in answer_lower for word in ['assessment', 'criteria', 'evaluation']):
            candidates = [
                "How is this evaluated?",
                "What are the scoring criteria?",
                "What determines success?",
            ]
        elif any(word in answer_lower for word in ['team', 'student', 'group']):
            candidates = [
                "What are the team responsibilities?",
                "How should teams collaborate?",
                "What resources are available for teams?",
            ]
        else:
            candidates = [
                "Can you provide more details about this?",
                "What are the practical applications?",
                "What should I know to get started?",
            ]
        
        for q in candidates:
            if q not in asked_questions and len(followups) < 3:
                followups.append(q)
    
    # Strategy 3: Question-type based follow-ups
    if len(followups) < 3:
        if any(word in question_lower for word in ['what is', 'define', 'explain', 'describe']):
            candidates = [
                "Can you give a practical example?",
                "How is this applied in real scenarios?",
                "What are the benefits of this?",
            ]
        elif any(word in question_lower for word in ['how', 'process', 'implement']):
            candidates = [
                "What are common mistakes to avoid?",
                "What tools or resources help with this?",
                "What are best practices?",
            ]
        elif any(word in question_lower for word in ['why', 'reason', 'purpose']):
            candidates = [
                "What are the implications?",
                "How does this compare to alternatives?",
                "What are the trade-offs?",
            ]
        else:
            candidates = [
                "What else should I know about this topic?",
                "Are there any prerequisites?",
                "What are related concepts?",
            ]
        
        for q in candidates:
            if q not in asked_questions and len(followups) < 3:
                followups.append(q)
    
    # Ensure we have unique questions
    followups = list(dict.fromkeys(followups))[:3]
    
    # If still empty, provide generic but useful questions
    if not followups:
        generic_options = [
            "What are the key takeaways?",
            "Can you elaborate further?",
            "What related information is available?",
        ]
        followups = [q for q in generic_options if q not in asked_questions][:3]
    
    return followups


def extract_topic_from_question(question):
    """
    Extract the main topic/subject from a user's question.
    
    Examples:
    "What is education?" -> "education"
    "How does RAG work?" -> "RAG"
    "Tell me about vector databases" -> "vector databases"
    """
    question_lower = question.lower().strip()
    
    # Remove question words and common phrases
    remove_patterns = [
        r'^what (is|are|does|do) (the |a |an )?',
        r'^how (does|do|is|are) (the |a |an )?',
        r'^why (is|are|does|do) (the |a |an )?',
        r'^when (is|are|does|do) (the |a |an )?',
        r'^where (is|are|does|do) (the |a |an )?',
        r'^who (is|are|does|do) (the |a |an )?',
        r'^can you (tell me about|explain|describe) (the |a |an )?',
        r'^tell me about (the |a |an )?',
        r'^explain (the |a |an )?',
        r'^describe (the |a |an )?',
    ]
    
    topic = question_lower
    for pattern in remove_patterns:
        topic = re.sub(pattern, '', topic, flags=re.IGNORECASE)
    
    # Remove trailing question marks and extra spaces
    topic = topic.rstrip('?').strip()
    
    # Get the first few meaningful words (up to 4 words)
    words = topic.split()[:4]
    
    # Remove common filler words from the topic
    filler_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'in', 'on', 'at', 'to', 'for', 'with', 'according'}
    meaningful_words = [w for w in words if w not in filler_words]
    
    if meaningful_words:
        return ' '.join(meaningful_words)
    elif words:
        return ' '.join(words)
    else:
        return None


def extract_meaningful_entities(text, original_question):
    """
    Extract MEANINGFUL entities from text, filtering out generic/meta terms.
    
    This is the KEY improvement - we filter out terms like "document", "file", "pdf"
    that are meta-references and don't represent actual content topics.
    """
    entities = []
    
    # First, try to get the topic from the original question
    question_topic = extract_topic_from_question(original_question)
    if question_topic:
        # Check if this topic appears in the answer
        if question_topic.lower() in text.lower():
            entities.append(question_topic)
    
    # IMPORTANT: Blacklist of generic/meta terms we should NEVER use as entities
    blacklist = {
        'document', 'documents', 'file', 'files', 'pdf', 'pdfs', 'text', 'content',
        'information', 'data', 'section', 'page', 'paragraph', 'sentence',
        'answer', 'question', 'query', 'response', 'result', 'output',
        'system', 'chatbot', 'model', 'context', 'source', 'index',
        'table', 'contents', 'introduction', 'conclusion', 'summary',
    }
    
    # Extract capitalized proper nouns (likely important terms)
    # But only if they're not in the blacklist
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for cap in capitalized:
        if cap.lower() not in blacklist and len(cap) > 2:
            entities.append(cap)
    
    # Look for domain-specific technical terms (whitelist approach)
    # These are terms that ARE meaningful for your project
    meaningful_patterns = [
        # AI/ML specific
        r'\b(rag|retrieval-augmented generation|vector database|embedding|llm|large language model)\b',
        r'\b(granite|langchain|chroma|faiss|milvus)\b',
        r'\b(machine learning|artificial intelligence|deep learning|neural network)\b',
        
        # Project-specific
        r'\b(capstone|project|implementation|development|pipeline)\b',
        r'\b(phase \d+|week \d+|sprint|milestone|deliverable)\b',
        
        # Academic/Education
        r'\b(education|learning|teaching|curriculum|pedagogy|assessment)\b',
        r'\b(student|instructor|professor|course|class|lecture)\b',
        
        # Technical processes
        r'\b(chunking|indexing|retrieval|generation|processing)\b',
        r'\b(api|interface|framework|architecture|component)\b',
    ]
    
    text_lower = text.lower()
    for pattern in meaningful_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if match.lower() not in blacklist:
                entities.append(match)
    
    # Look for quoted terms (often important concepts)
    quoted = re.findall(r'"([^"]+)"', text)
    for q in quoted:
        if q.lower() not in blacklist and len(q.split()) <= 3:
            entities.append(q)
    
    # Deduplicate and return top 5 unique entities
    unique_entities = []
    seen = set()
    for entity in entities:
        entity_lower = entity.lower()
        if entity_lower not in seen and entity_lower not in blacklist:
            seen.add(entity_lower)
            unique_entities.append(entity)
    
    return unique_entities[:5]


# ----------------------------
# Helper: build QA chain
# ----------------------------
def build_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

    llm = WatsonxLLM(
        url="https://us-south.ml.cloud.ibm.com",
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("IBM_PROJECT_ID"),
        model_id="ibm/granite-3-8b-instruct",
        params={
            "temperature": 0.1,
            "max_new_tokens": 200,
            "repetition_penalty": 1.1,
        },
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
            }
        ),
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )

    return qa_chain


def process_question(qa_chain, prompt, is_followup=False):
    """
    Process questions with better prompt handling.
    
    Args:
        qa_chain: The QA chain
        prompt: User's question
        is_followup: Whether this is a follow-up question
    
    Returns:
        dict with answer, sources, and whether it succeeded
    """
    try:
        # Different prompting strategy for follow-up questions
        if is_followup:
            styled_question = (
                f"{prompt}\n\n"
                "Provide a focused answer in 2-3 bullet points based on the document content."
            )
        else:
            styled_question = (
                f"{prompt}\n\n"
                "Give a brief answer in 2-3 bullet points using information from the document."
            )

        response = qa_chain.invoke({"question": styled_question})
        raw_answer = response.get("answer", "").strip()
        
        print(f"DEBUG - Question type: {'followup' if is_followup else 'initial'}")
        print(f"DEBUG - Raw answer: '{raw_answer}'")

        # Improved answer cleaning
        answer = clean_answer(raw_answer)
        
        if not answer or len(answer) < 10:
            print(f"WARNING - Answer too short, using raw: '{raw_answer}'")
            answer = raw_answer
        
        if not answer or len(answer) < 5:
            answer = "⚠️ I couldn't generate a proper answer. Please try rephrasing your question."
            print(f"ERROR - Empty answer. Raw was: '{raw_answer}'")

        # Process sources
        raw_sources = []
        for doc in response.get("source_documents", []):
            src_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content.strip(),
            }
            raw_sources.append(src_info)

        # Deduplicate sources
        sources = []
        seen = set()
        for s in raw_sources:
            key = (s["source"], str(s["page"]))
            if key not in seen:
                seen.add(key)
                sources.append(s)

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
        }
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"EXCEPTION: {e}")
        return {
            "success": False,
            "answer": error_msg,
            "sources": [],
        }


def clean_answer(raw_answer):
    """
    Clean up the answer by removing prompt echoes and formatting issues.
    """
    answer = raw_answer
    
    # Remove "Answer:" prefix if present
    if "Answer:" in answer:
        parts = answer.split("Answer:")
        if len(parts) > 1:
            answer = parts[-1].strip()
    
    # Remove "Question:" prefix if present
    if answer.lower().startswith("question:"):
        lines = answer.split('\n', 1)
        if len(lines) > 1:
            answer = lines[1].strip()
    
    # Remove any remaining prompt artifacts
    artifacts = [
        "Give a brief answer in 2-3 bullet points",
        "using information from the document",
        "Provide a focused answer",
        "based on the document content",
    ]
    
    for artifact in artifacts:
        answer = answer.replace(artifact, "")
    
    return answer.strip()


# ----------------------------
# Streamlit app
# ----------------------------
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Document Q&A Chatbot")
st.markdown("Ask me anything about your documents!")


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    with st.spinner("🔄 Loading AI system..."):
        st.session_state.qa_chain = build_qa_chain()
    st.success("✅ System ready!")

# Track which follow-up button was clicked
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ----------------------------
# Display chat history
# ----------------------------
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(message["sources"], 1):
                    filename = os.path.basename(src["source"])
                    st.write(f"**Source {i} – {filename}, page {src['page']}**")
                    st.write(src["content"])
                    st.markdown("---")
        
        # Display follow-up questions with working buttons
        if "followup_questions" in message and message["followup_questions"]:
            st.markdown("---")
            st.markdown("**💡 You might also want to ask:**")
            
            for q_idx, followup_q in enumerate(message["followup_questions"]):
                # Use a unique key based on message index and question index
                button_key = f"followup_{idx}_{q_idx}"
                if st.button(followup_q, key=button_key, use_container_width=True):
                    st.session_state.pending_question = followup_q
                    st.rerun()


# ----------------------------
# Process pending follow-up question
# ----------------------------
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None  # Clear it
    
    # Add to messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process the question (MARKED AS FOLLOW-UP)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = process_question(
                st.session_state.qa_chain, 
                prompt, 
                is_followup=True
            )
            
            if result["success"]:
                answer = result["answer"]
                sources = result["sources"]
                
                # Generate NEW follow-up questions (passing conversation history)
                followup_questions = generate_followup_questions(
                    answer, 
                    prompt,
                    conversation_history=st.session_state.messages
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources,
                        "followup_questions": followup_questions
                    }
                )
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )
    
    st.rerun()


# ----------------------------
# User input + Assistant response
# ----------------------------
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = process_question(
                st.session_state.qa_chain, 
                prompt, 
                is_followup=False
            )
            
            if result["success"]:
                answer = result["answer"]
                sources = result["sources"]
                
                st.markdown(f"### 🟢 Answer\n{answer}")

                if sources:
                    with st.expander("📚 View Sources"):
                        for i, src in enumerate(sources, 1):
                            filename = os.path.basename(src["source"])
                            st.markdown(f"#### 📄 Source {i}")
                            st.write(f"**File:** {filename}")
                            st.write(f"**Page:** {src['page']}")
                            st.write(src["content"])
                            st.markdown("---")

                # Generate follow-up questions (with conversation history)
                followup_questions = generate_followup_questions(
                    answer, 
                    prompt,
                    conversation_history=st.session_state.messages
                )
                
                if followup_questions:
                    st.markdown("---")
                    st.markdown("**💡 You might also want to ask:**")
                    for q_idx, followup_q in enumerate(followup_questions):
                        button_key = f"followup_new_{len(st.session_state.messages)}_{q_idx}"
                        if st.button(followup_q, key=button_key, use_container_width=True):
                            st.session_state.pending_question = followup_q
                            st.rerun()

                st.session_state.messages.append(
                    {
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources,
                        "followup_questions": followup_questions
                    }
                )
            else:
                st.error(result["answer"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown(
        """
        This chatbot uses:
        - **IBM Granite 3 LLM**
        - **LangChain Conversational RAG**
        - **Chroma Vector DB**
        - **Smart Context-Aware Follow-ups**

        Upload a PDF and ask questions about its contents!
        """
    )

    st.markdown("---")

    st.header("📊 Chat Statistics")
    st.write(f"**Messages:** {len(st.session_state.messages)}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.qa_chain.memory.clear()
        st.rerun()

    st.markdown("---")

    st.header("📄 Upload a PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


# ----------------------------
# File upload handler
# ----------------------------
if uploaded_file:
    st.write("📄 Processing uploaded PDF...")

    os.makedirs("./input", exist_ok=True)
    file_path = os.path.join("./input", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    query_util.process_pdf(file_path)

    st.session_state.qa_chain = build_qa_chain()

    st.success("✅ PDF added successfully! You can now ask questions.")