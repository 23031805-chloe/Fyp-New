import os
import re
import json
import uuid
import shutil
import sqlite3
import csv
import datetime
import hashlib
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st
from dotenv import load_dotenv

import query_util

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ibm import WatsonxLLM


# ==================================================
# Config (ONLY ONCE)
# ==================================================
load_dotenv()
st.set_page_config(page_title="Document Q&A Chatbot", page_icon="🤖", layout="wide")


# ==================================================
# Wahyu: Follow-up question generation (KEEP)
# ==================================================
def extract_topic_from_question(question):
    question_lower = (question or "").lower().strip()

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

    topic = topic.rstrip('?').strip()

    words = topic.split()[:4]
    filler_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'in', 'on', 'at', 'to', 'for', 'with', 'according'}
    meaningful_words = [w for w in words if w not in filler_words]

    if meaningful_words:
        return ' '.join(meaningful_words)
    elif words:
        return ' '.join(words)
    else:
        return None


def extract_meaningful_entities(text, original_question):
    entities = []

    question_topic = extract_topic_from_question(original_question)
    if question_topic and question_topic.lower() in (text or "").lower():
        entities.append(question_topic)

    blacklist = {
        'document', 'documents', 'file', 'files', 'pdf', 'pdfs', 'text', 'content',
        'information', 'data', 'section', 'page', 'paragraph', 'sentence',
        'answer', 'question', 'query', 'response', 'result', 'output',
        'system', 'chatbot', 'model', 'context', 'source', 'index',
        'table', 'contents', 'introduction', 'conclusion', 'summary',
    }

    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text or "")
    for cap in capitalized:
        if cap.lower() not in blacklist and len(cap) > 2:
            entities.append(cap)

    meaningful_patterns = [
        r'\b(rag|retrieval-augmented generation|vector database|embedding|llm|large language model)\b',
        r'\b(granite|langchain|chroma|faiss|milvus)\b',
        r'\b(machine learning|artificial intelligence|deep learning|neural network)\b',
        r'\b(capstone|project|implementation|development|pipeline)\b',
        r'\b(phase \d+|week \d+|sprint|milestone|deliverable)\b',
        r'\b(education|learning|teaching|curriculum|pedagogy|assessment)\b',
        r'\b(student|instructor|professor|course|class|lecture)\b',
        r'\b(chunking|indexing|retrieval|generation|processing)\b',
        r'\b(api|interface|framework|architecture|component)\b',
    ]

    text_lower = (text or "").lower()
    for pattern in meaningful_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            m = match if isinstance(match, str) else " ".join(match)
            if m.lower() not in blacklist:
                entities.append(m)

    quoted = re.findall(r'"([^"]+)"', text or "")
    for q in quoted:
        if q.lower() not in blacklist and len(q.split()) <= 3:
            entities.append(q)

    unique_entities = []
    seen = set()
    for entity in entities:
        el = entity.lower()
        if el not in seen and el not in blacklist:
            seen.add(el)
            unique_entities.append(entity)

    return unique_entities[:5]


def generate_followup_questions(answer, original_question, conversation_history=None):
    followups = []
    answer_lower = (answer or "").lower()
    question_lower = (original_question or "").lower()

    asked_questions = set()
    if conversation_history:
        for msg in conversation_history:
            if msg.get("role") == "assistant" and "followups" in msg:
                asked_questions.update(msg["followups"])

    entities = extract_meaningful_entities(answer or "", original_question or "")

    if any(phrase in answer_lower for phrase in [
        "does not provide", "cannot find", "no information",
        "not mentioned", "not specified", "would be required"
    ]):
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

    if entities:
        entity = entities[0]
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

    followups = list(dict.fromkeys(followups))[:3]
    if not followups:
        generic_options = [
            "What are the key takeaways?",
            "Can you elaborate further?",
            "What related information is available?",
        ]
        followups = [q for q in generic_options if q not in asked_questions][:3]

    return followups


# ==================================================
# Feedback (Xinru) — kept (optional)
# ==================================================
def save_feedback(query, response, rating, comment=""):
    try:
        file_exists = os.path.isfile('feedback.csv')
        with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Query', 'Response', 'Rating', 'Comment'])
            writer.writerow([datetime.datetime.now(), query, response, rating, comment])
    except PermissionError:
        st.error("⚠️ Could not save feedback! Please close 'feedback.csv'.")


# ==================================================
# Login + Email verification (Kaixin) — kept
# ==================================================
def load_css():
    st.markdown("""
        <style>
            .stApp { background-color: #f8f9fa; }
            .css-1r6slb0, .css-12oz5g7 { max-width: 450px; padding-top: 5rem; }
            div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
                background-color: #ffffff;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08);
                border: 1px solid #f0f0f0;
            }
            .stTextInput input {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px;
                color: #333;
            }
            .stTextInput input:focus {
                border-color: #2563eb;
                box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
            }
            .stButton button {
                width: 100%;
                background-color: #2563eb !important;
                color: #ffffff !important;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.2s;
            }
            .stButton button p { color: #ffffff !important; }
            .stButton button:hover {
                background-color: #1d4ed8 !important;
                transform: translateY(-1px);
            }
            .stTabs [data-baseweb="tab"] {
                color: #64748b; font-weight: 500; border: none; background-color: transparent;
            }
            .stTabs [aria-selected="true"] {
                color: #2563eb !important;
                border-bottom-color: #2563eb !important;
                border-bottom-width: 3px !important;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 20px; border-bottom: 1px solid #f0f0f0; margin-bottom: 25px;
            }
            h2 { color: #1e293b; font-weight: 700; letter-spacing: -0.5px; }
            p { color: #64748b; }
        </style>
    """, unsafe_allow_html=True)


def is_valid_email(email):
    email = (email or "").strip().lower()
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email)


def is_valid_password(password):
    if len(password) < 12:
        return False, "⚠️ Password must be at least 12 characters."
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&_])[A-Za-z\d@$!%*?&_]{12,}$"
    if not re.match(pattern, password or ""):
        return False, "⚠️ Must have Uppercase, Lowercase, Number & Symbol (@$!%*?&_)."
    return True, "Valid"


def send_verification_email(to_email, code):
    sender_email = os.getenv("GMAIL_USER")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender_email or not sender_password:
        return False, "⚠️ Setup Error: Missing GMAIL_USER or GMAIL_APP_PASSWORD in .env"

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = "Your Verification Code"

        body = f"""
        <html><body>
            <h2>Verification Code</h2>
            <p>Your secure code is:</p>
            <h1 style="color: #2563eb;">{code}</h1>
        </body></html>
        """
        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        return True, "✅ Email Sent!"
    except Exception as e:
        return False, f"❌ Email Failed: {str(e)}"


def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password_hash TEXT
        )
    ''')
    conn.commit()
    conn.close()


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text


def create_user(username, email, password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users VALUES (?, ?, ?)',
                  (username, email, make_hashes(password)))
        conn.commit()
        conn.close()
        return True, "✅ Account created! Please Login."
    except sqlite3.IntegrityError:
        return False, "⚠️ Username or Email already exists."
    except Exception as e:
        return False, f"Error: {e}"


def check_login(username, password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
        data = c.fetchone()
        conn.close()
        if data and check_hashes(password, data[0]):
            return True
        return False
    except Exception:
        return False


def check_email_exists(email):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT username FROM users WHERE email = ?', (email,))
        data = c.fetchone()
        conn.close()
        return data[0] if data else None
    except Exception:
        return None


def reset_password_in_db(email, new_password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('UPDATE users SET password_hash = ? WHERE email = ?',
                  (make_hashes(new_password), email))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def login_signup_page():
    load_css()
    st.title("Document Retrieval System")

    tab1, tab2, tab3 = st.tabs(["Log In", "Sign Up", "Recover"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign In"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.toast("Welcome back!", icon="👋")
                st.rerun()
            else:
                st.error("❌ Incorrect username or password")

    with tab2:
        if "signup_stage" not in st.session_state:
            st.session_state.signup_stage = 1

        if st.session_state.signup_stage == 1:
            new_user = st.text_input("Username", key="new_user")
            new_email = st.text_input("Email", key="new_email")
            new_pass = st.text_input("Password", type="password", key="new_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")

            if st.button("Verify Email & Create"):
                valid_p, msg = is_valid_password(new_pass)
                if not new_user:
                    st.warning("Username missing")
                elif not is_valid_email(new_email):
                    st.warning("Invalid Email")
                elif not valid_p:
                    st.warning(msg)
                elif new_pass != confirm_pass:
                    st.error("Passwords don't match")
                elif check_email_exists(new_email):
                    st.error("Email already registered!")
                else:
                    code = ''.join(random.choices(string.digits, k=6))
                    success, email_msg = send_verification_email(new_email, code)
                    if success:
                        st.session_state.signup_otp = code
                        st.session_state.signup_data = (new_user, new_email, new_pass)
                        st.session_state.signup_stage = 2
                        st.toast("📩 Check your inbox!", icon="✅")
                        st.rerun()
                    else:
                        st.error(email_msg)

        elif st.session_state.signup_stage == 2:
            st.info(f"Code sent to {st.session_state.signup_data[1]}")
            otp_input = st.text_input("Enter Email Code", key="signup_otp_code")

            if st.button("Confirm & Register"):
                if otp_input == st.session_state.signup_otp:
                    u, e, p = st.session_state.signup_data
                    success, db_msg = create_user(u, e, p)
                    if success:
                        st.success(db_msg)
                        st.session_state.signup_stage = 1
                    else:
                        st.error(db_msg)
                else:
                    st.error("❌ Invalid Code")

            if st.button("Back"):
                st.session_state.signup_stage = 1
                st.rerun()

    with tab3:
        if "reset_stage" not in st.session_state:
            st.session_state.reset_stage = 1

        if st.session_state.reset_stage == 1:
            reset_email = st.text_input("Enter Registered Email", key="reset_email_input")
            if st.button("Send Reset Code"):
                if check_email_exists(reset_email):
                    code = ''.join(random.choices(string.digits, k=6))
                    success, email_msg = send_verification_email(reset_email, code)
                    if success:
                        st.session_state.reset_otp = code
                        st.session_state.reset_email = reset_email
                        st.session_state.reset_stage = 2
                        st.toast("📩 Code sent!", icon="✅")
                        st.rerun()
                    else:
                        st.error(email_msg)
                else:
                    st.error("Email not found.")

        elif st.session_state.reset_stage == 2:
            st.info(f"Check {st.session_state.reset_email} for code")
            otp_input = st.text_input("Enter Code", key="reset_otp_input")
            new_p = st.text_input("New Password", type="password", key="reset_new_p")

            if st.button("Change Password"):
                if otp_input == st.session_state.reset_otp:
                    valid, msg = is_valid_password(new_p)
                    if valid:
                        if reset_password_in_db(st.session_state.reset_email, new_p):
                            st.success("✅ Password Changed! Please Log In.")
                            st.session_state.reset_stage = 1
                        else:
                            st.error("DB Error")
                    else:
                        st.warning(msg)
                else:
                    st.error("Invalid Code")


# ==================================================
# Projects + Chats (You) — kept
# ==================================================
BASE_PROJECT_DIR = "./projects"
GLOBAL_DIR = "./global"
os.makedirs(BASE_PROJECT_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)
GLOBAL_CHATS_FILE = os.path.join(GLOBAL_DIR, "chats.json")


def _safe_load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _safe_save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def project_paths(project_name: str):
    base = os.path.join(BASE_PROJECT_DIR, project_name)
    return {
        "base": base,
        "input": os.path.join(base, "input"),
        "chroma": os.path.join(base, "chroma_db"),
        "chats": os.path.join(base, "chats.json"),
    }


def init_project(project_name: str):
    paths = project_paths(project_name)
    os.makedirs(paths["input"], exist_ok=True)
    os.makedirs(paths["chroma"], exist_ok=True)
    if not os.path.exists(paths["chats"]):
        _safe_save_json(paths["chats"], {"chats": []})


def _normalize_project_name(name: str) -> str:
    name = (name or "").strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = " ".join(name.split())
    return name


def rename_project(old_name: str, new_name: str):
    old_name = _normalize_project_name(old_name)
    new_name = _normalize_project_name(new_name)

    if not old_name or not new_name:
        return False, "Project name cannot be empty."
    if old_name == "Default":
        return False, "You cannot rename the Default project."
    if old_name == new_name:
        return False, "New name is the same as current name."

    old_base = project_paths(old_name)["base"]
    new_base = project_paths(new_name)["base"]

    if not os.path.exists(old_base):
        return False, "Project not found."
    if os.path.exists(new_base):
        return False, "A project with that name already exists."

    try:
        shutil.move(old_base, new_base)
        init_project(new_name)
        return True, "Project renamed."
    except Exception as e:
        return False, f"Rename failed: {e}"


def delete_project(name: str):
    name = _normalize_project_name(name)

    if not name:
        return False, "Project name cannot be empty."
    if name == "Default":
        return False, "You cannot delete the Default project."

    base = project_paths(name)["base"]
    if not os.path.exists(base):
        return False, "Project not found."

    try:
        shutil.rmtree(base)
        return True, "Project deleted."
    except Exception as e:
        return False, f"Delete failed: {e}"


def list_projects():
    init_project("Default")
    projs = ["Default"]
    for d in os.listdir(BASE_PROJECT_DIR):
        p = os.path.join(BASE_PROJECT_DIR, d)
        if os.path.isdir(p) and d != "Default":
            projs.append(d)
    return sorted(set(projs), key=lambda x: (x != "Default", x.lower()))


def load_project_chats(project_name: str):
    init_project(project_name)
    data = _safe_load_json(project_paths(project_name)["chats"], {"chats": []})
    if "chats" not in data or not isinstance(data["chats"], list):
        data = {"chats": []}
    return data


def save_project_chats(project_name: str, data):
    _safe_save_json(project_paths(project_name)["chats"], data)


def load_global_chats():
    if not os.path.exists(GLOBAL_CHATS_FILE):
        _safe_save_json(GLOBAL_CHATS_FILE, {"chats": []})
    data = _safe_load_json(GLOBAL_CHATS_FILE, {"chats": []})
    if "chats" not in data or not isinstance(data["chats"], list):
        data = {"chats": []}
    return data


def save_global_chats(data):
    _safe_save_json(GLOBAL_CHATS_FILE, data)


def make_new_chat():
    return {"id": str(uuid.uuid4()), "title": "New chat", "messages": []}


def auto_title_from_prompt(prompt: str):
    line = (prompt or "").strip().split("\n")[0].strip()
    if not line:
        return "New chat"
    return (line[:40] + "…") if len(line) > 40 else line


def get_uploaded_files_for_project(project_name: str):
    paths = project_paths(project_name)
    if not os.path.exists(paths["input"]):
        return []
    files = [f for f in os.listdir(paths["input"]) if os.path.isfile(os.path.join(paths["input"], f))]
    return sorted(files, key=lambda x: x.lower())


# ==================================================
# LangChain / RAG (shared)
# ==================================================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return WatsonxLLM(
        url="https://us-south.ml.cloud.ibm.com",
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("IBM_PROJECT_ID"),
        model_id="ibm/granite-3-8b-instruct",
        params={"temperature": 0.1, "max_new_tokens": 512, "repetition_penalty": 1.1},
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore(persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())


def build_chain(persist_dir: str, memory: ConversationBufferMemory):
    vectorstore = get_vectorstore(persist_dir)
    llm = get_llm()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )


def safe_process_document(path: str, persist_dir: str):
    """
    Calls your query_util indexing function safely.
    Adjust here if your function name differs.
    """
    if hasattr(query_util, "process_document"):
        query_util.process_document(path, persist_directory=persist_dir)
        return
    # Fallback (if your util uses process_pdf instead)
    if hasattr(query_util, "process_pdf"):
        query_util.process_pdf(path)
        return
    raise AttributeError("query_util needs process_document(...) or process_pdf(...)")


# ==================================================
# Session state
# ==================================================
init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "current_project" not in st.session_state:
    st.session_state.current_project = "Default"
if "active_scope" not in st.session_state:
    st.session_state.active_scope = "global"  # "global" or "project"
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# for follow-up button clicks
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

# memory per chat
if "chat_memories" not in st.session_state:
    st.session_state.chat_memories = {}  # key: f"{scope}:{chat_id}" -> ConversationBufferMemory


# ==================================================
# Login gate
# ==================================================
if not st.session_state.logged_in:
    login_signup_page()
    st.stop()


# ==================================================
# Load chats
# ==================================================
projects = list_projects()
if st.session_state.current_project not in projects:
    st.session_state.current_project = "Default"

project_name = st.session_state.current_project
project_data = load_project_chats(project_name)
global_data = load_global_chats()

project_chats = project_data["chats"]
global_chats = global_data["chats"]


def find_chat(scope: str, chat_id: str):
    chats = project_chats if scope == "project" else global_chats
    for c in chats:
        if c["id"] == chat_id:
            return c
    return None


def ensure_active_chat():
    scope = st.session_state.active_scope
    chats = project_chats if scope == "project" else global_chats

    if st.session_state.active_chat_id and find_chat(scope, st.session_state.active_chat_id):
        return

    if chats:
        st.session_state.active_chat_id = chats[-1]["id"]
    else:
        new_chat = make_new_chat()
        chats.append(new_chat)
        st.session_state.active_chat_id = new_chat["id"]
        if scope == "project":
            save_project_chats(project_name, project_data)
        else:
            save_global_chats(global_data)


ensure_active_chat()
current_chat = find_chat(st.session_state.active_scope, st.session_state.active_chat_id)


# ==================================================
# Sidebar (projects + chats + upload)
# ==================================================
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state.current_user}")

    if st.button("🚪 Log out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

    st.markdown("---")

    if st.button("✏️ New chat", use_container_width=True):
        new_chat = make_new_chat()
        global_chats.append(new_chat)
        st.session_state.active_scope = "global"
        st.session_state.active_chat_id = new_chat["id"]
        save_global_chats(global_data)
        st.rerun()

    st.text_input("Search chats", key="search_query", placeholder="Search chats…")
    st.markdown("---")

    st.markdown("### Projects")
    selected_project = st.selectbox("Select project", projects, index=projects.index(project_name))
    if selected_project != project_name:
        st.session_state.current_project = selected_project
        st.session_state.active_chat_id = None
        st.rerun()

    new_project_name = st.text_input("New project name", placeholder="e.g. FYP Chatbot")
    if st.button("New project", use_container_width=True):
        name = _normalize_project_name(new_project_name)
        if name:
            init_project(name)
            st.session_state.current_project = name
            st.session_state.active_scope = "project"
            st.session_state.active_chat_id = None
            st.rerun()

    st.markdown("#### Manage project")
    with st.expander("✏️ Rename project"):
        if project_name == "Default":
            st.info("Default project cannot be renamed.")
        else:
            rename_to = st.text_input("Rename to", key="rename_project_to", placeholder="e.g. Capstone Chatbot")
            if st.button("Rename", key="btn_rename_project", use_container_width=True):
                ok, msg = rename_project(project_name, rename_to)
                if ok:
                    st.success(msg)
                    st.session_state.current_project = _normalize_project_name(rename_to)
                    st.session_state.active_scope = "project"
                    st.session_state.active_chat_id = None
                    st.rerun()
                else:
                    st.error(msg)

    with st.expander("🗑️ Delete project"):
        if project_name == "Default":
            st.info("Default project cannot be deleted.")
        else:
            st.warning("This deletes the project folder (documents, chats, vector DB).")
            confirm = st.checkbox(f"I understand. Delete '{project_name}' permanently.", key="confirm_delete_project")
            if st.button("Delete", key="btn_delete_project", use_container_width=True, disabled=not confirm):
                ok, msg = delete_project(project_name)
                if ok:
                    st.success(msg)
                    st.session_state.current_project = "Default"
                    st.session_state.active_scope = "global"
                    st.session_state.active_chat_id = None
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("---")
    st.markdown(f"### {project_name} chats")
    if st.button("New project chat", use_container_width=True):
        new_chat = make_new_chat()
        project_chats.append(new_chat)
        st.session_state.active_scope = "project"
        st.session_state.active_chat_id = new_chat["id"]
        save_project_chats(project_name, project_data)
        st.rerun()

    def render_chat_list(chats, scope: str):
        q = (st.session_state.search_query or "").strip().lower()
        filtered = []
        for c in chats:
            title = (c.get("title") or "New chat")
            if not q or q in title.lower():
                filtered.append(c)

        for c in reversed(filtered):
            is_active = (st.session_state.active_scope == scope and st.session_state.active_chat_id == c["id"])
            label = c.get("title") or "New chat"

            cols = st.columns([0.84, 0.16])
            if cols[0].button(("✅ " + label) if is_active else label, key=f"open_{scope}_{c['id']}", use_container_width=True):
                st.session_state.active_scope = scope
                st.session_state.active_chat_id = c["id"]
                st.rerun()

            with cols[1].popover("⋯", use_container_width=True):
                if st.button("Delete", key=f"del_{scope}_{c['id']}", use_container_width=True):
                    if scope == "project":
                        project_data["chats"] = [x for x in project_data["chats"] if x["id"] != c["id"]]
                        save_project_chats(project_name, project_data)
                        st.session_state.active_scope = "project"
                    else:
                        global_data["chats"] = [x for x in global_data["chats"] if x["id"] != c["id"]]
                        save_global_chats(global_data)
                        st.session_state.active_scope = "global"

                    st.session_state.active_chat_id = None
                    st.rerun()

    render_chat_list(project_chats, "project")
    st.markdown("---")
    st.markdown("### Your chats")
    render_chat_list(global_chats, "global")

    st.markdown("---")
    st.markdown("### 📎 Uploaded documents")
    uploaded_files = get_uploaded_files_for_project(project_name)
    if uploaded_files:
        for f in uploaded_files:
            st.write(f"• {f}")
    else:
        st.write("No documents uploaded yet.")

    uploaded = st.file_uploader("Upload PDF / Word / Text", type=["pdf", "docx", "txt"])


# ==================================================
# File processing (PROJECT KB)
# ==================================================
if uploaded:
    init_project(project_name)
    paths = project_paths(project_name)

    file_path = os.path.join(paths["input"], uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("Indexing document…"):
        safe_process_document(file_path, persist_dir=paths["chroma"])

    try:
        get_vectorstore.clear()
    except Exception:
        pass

    st.toast("📄 Document indexed!", icon="✅")
    st.rerun()


# ==================================================
# Main UI
# ==================================================
st.title("🤖 Document Q&A Chatbot")
st.markdown("Ask me anything about your documents!")

if current_chat is None:
    st.info("No chat selected.")
    st.stop()


def chat_key(scope: str, chat_id: str) -> str:
    return f"{scope}:{chat_id}"


def get_memory_for_chat(scope: str, chat_id: str) -> ConversationBufferMemory:
    key = chat_key(scope, chat_id)
    if key not in st.session_state.chat_memories:
        st.session_state.chat_memories[key] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
    return st.session_state.chat_memories[key]


def render_sources(docs):
    if not docs:
        return
    with st.expander("📚 View Sources"):
        seen = set()
        idx = 1
        for doc in docs:
            file = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            key = (file, str(page))
            if key in seen:
                continue
            seen.add(key)

            st.subheader(f"📄 Source {idx}")
            st.write(f"**File:** {os.path.basename(file)}")
            st.write(f"**Page:** {page}")
            text = (doc.page_content or "").strip()
            st.write(text[:1500] + ("…" if len(text) > 1500 else ""))
            st.markdown("---")
            idx += 1


# Render messages + follow-up buttons
for mi, msg in enumerate(current_chat["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            render_sources(msg["sources"])

        if msg.get("followups"):
            st.markdown("---")
            st.markdown("**💡 You might also want to ask:**")
            for qi, fq in enumerate(msg["followups"]):
                if st.button(fq, key=f"fu_{current_chat['id']}_{mi}_{qi}", use_container_width=True):
                    st.session_state.queued_prompt = fq
                    st.rerun()


# Input (supports follow-up click -> queued_prompt)
prompt = None
if st.session_state.queued_prompt:
    prompt = st.session_state.queued_prompt
    st.session_state.queued_prompt = None
else:
    prompt = st.chat_input("Ask a question about your documents…")

if prompt:
    if len(current_chat["messages"]) == 0:
        current_chat["title"] = auto_title_from_prompt(prompt)

    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            docs = []
            answer = ""
            scope = st.session_state.active_scope

            try:
                if scope == "project":
                    paths = project_paths(project_name)
                    init_project(project_name)
                    memory = get_memory_for_chat(scope, current_chat["id"])
                    chain = build_chain(paths["chroma"], memory=memory)
                    response = chain({"question": prompt})
                    answer = response.get("answer", "⚠ No answer returned.")
                    docs = response.get("source_documents", []) or []
                else:
                    llm = get_llm()
                    answer = llm.invoke(prompt)

            except Exception as e:
                answer = "❌ Error generating answer. Check your API keys / Watsonx configuration."
                st.error(str(e))

            st.markdown(answer)

            if scope == "project" and docs:
                render_sources(docs)

            followups = generate_followup_questions(
                answer,
                prompt,
                conversation_history=current_chat["messages"]
            )

            if followups:
                st.markdown("---")
                st.markdown("**💡 You might also want to ask:**")
                for qi, fq in enumerate(followups):
                    if st.button(fq, key=f"fu_live_{current_chat['id']}_{qi}", use_container_width=True):
                        st.session_state.queued_prompt = fq
                        st.rerun()

    # store assistant message + citations + followups
    current_chat["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": docs if st.session_state.active_scope == "project" else [],
            "followups": followups,
        }
    )

    # Persist to JSON
    if st.session_state.active_scope == "project":
        save_project_chats(project_name, project_data)
    else:
        save_global_chats(global_data)

    st.rerun()
