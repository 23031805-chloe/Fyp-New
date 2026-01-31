import os
import json
import uuid
import shutil
import streamlit as st
from dotenv import load_dotenv
import query_util
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ibm import WatsonxLLM
import sqlite3
import csv
import datetime
import hashlib
import re
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# --- FEEDBACK FUNCTION ---
def save_feedback(query, response, rating, comment=""):
    try:
        file_exists = os.path.isfile('feedback.csv')
        with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Query', 'Response', 'Rating', 'Comment'])
            writer.writerow([datetime.datetime.now(), query, response, rating, comment])
    except PermissionError:
        st.error("ΓÜá∩╕Å Could not save feedback! Please close 'feedback.csv'.")

# --- CUSTOM CSS STYLING ---
def load_css():
    st.markdown("""
        <style>
            /* 1. Global Background */
            .stApp { background-color: #f8f9fa; }
            
            /* 2. Center the Login Card */
            .css-1r6slb0, .css-12oz5g7 { max-width: 450px; padding-top: 5rem; }
            
            /* 3. The Card Itself */
            div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
                background-color: #ffffff;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08); 
                border: 1px solid #f0f0f0;
            }

            /* 4. Input Fields */
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
            
            /* 5. Buttons */
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
            
            /* 6. Tabs - Blue Theme */
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

            /* 7. Typography */
            h2 { color: #1e293b; font-weight: 700; letter-spacing: -0.5px; }
            p { color: #64748b; }
        </style>
    """, unsafe_allow_html=True)

# --- VALIDATION HELPERS ---
def is_valid_email(email):
    email = (email or "").strip().lower()
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email)

def is_valid_password(password):
    if len(password) < 12:
        return False, "ΓÜá∩╕Å Password must be at least 12 characters."
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&_])[A-Za-z\d@$!%*?&_]{12,}$"
    if not re.match(pattern, password):
        return False, "ΓÜá∩╕Å Must have Uppercase, Lowercase, Number & Symbol (@$!%*?&_)."
    return True, "Valid"

def send_verification_email(to_email, code):
    sender_email = os.getenv("GMAIL_USER")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")
    
    if not sender_email or not sender_password:
        return False, "ΓÜá∩╕Å Setup Error: Missing GMAIL_USER or GMAIL_APP_PASSWORD in .env"

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = "≡ƒöÉ Your Verification Code"

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
        return True, "Γ£à Email Sent!"
    except Exception as e:
        return False, f"Γ¥î Email Failed: {str(e)}"

# --- DATABASE SETUP ---
def init_db():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        # New Table Structure: Username, Email, Password Hash
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                password_hash TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Init Error: {e}")

init_db()

# --- AUTH FUNCTIONS ---
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
        return True, "Γ£à Account created! Please Login."
    except sqlite3.IntegrityError:
        return False, "ΓÜá∩╕Å Username or Email already exists."
    except Exception as e:
        return False, f"Error: {e}"
    
def update_user_details(current_username, old_password, new_username, new_email, new_password=None):
    """Updates profile, but ONLY if the old password is correct."""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # 1. VERIFY OLD PASSWORD FIRST
        c.execute("SELECT password_hash FROM users WHERE username = ?", (current_username,))
        result = c.fetchone()
        if not result:
            conn.close()
            return False, "User not found."
            
        stored_hash = result[0]
        if not check_hashes(old_password, stored_hash):
            conn.close()
            return False, "Γ¥î Incorrect Current Password. Changes NOT saved."

        # 2. Check if new username/email is already taken by SOMEONE ELSE
        if new_username != current_username:
            c.execute("SELECT * FROM users WHERE username = ?", (new_username,))
            if c.fetchone(): 
                conn.close()
                return False, "Username already taken."
            
        if new_email:
            # Get current email to see if it changed
            c.execute("SELECT email FROM users WHERE username = ?", (current_username,))
            curr_email = c.fetchone()[0]
            if new_email != curr_email:
                c.execute("SELECT * FROM users WHERE email = ?", (new_email,))
                if c.fetchone(): 
                    conn.close()
                    return False, "Email already in use."

        # 3. Update the record
        if new_password:
            # Update everything including password
            c.execute('''UPDATE users SET username = ?, email = ?, password_hash = ? 
                         WHERE username = ?''', 
                      (new_username, new_email, make_hashes(new_password), current_username))
        else:
            # Update only profile info
            c.execute('''UPDATE users SET username = ?, email = ? 
                         WHERE username = ?''', 
                      (new_username, new_email, current_username))
            
        conn.commit()
        conn.close()
        return True, "Γ£à Profile updated! Please re-login."
    except Exception as e:
        return False, f"Error: {e}"

def delete_user_account(username):
    """Permanently deletes the user"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(e)
        return False

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
    except:
        return False

def check_email_exists(email):
    """Returns username if email exists, else None"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT username FROM users WHERE email = ?', (email,))
        data = c.fetchone()
        conn.close()
        return data[0] if data else None
    except:
        return None

def reset_password_in_db(email, new_password):
    """Updates the password for the given email"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('UPDATE users SET password_hash = ? WHERE email = ?', 
                 (make_hashes(new_password), email))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(e)
        return False

# --- LOGIN PAGE LOGIC ---
def login_signup_page():
    load_css()
    
    with st.container():
        col1, col2 = st.columns([1, 1])
        
        # LEFT: Marketing
        with col1:
            st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
            st.markdown("# Document Retrieval System")
            st.markdown("""
            ### Unlock the power of your Documents.
            This Intelligent Document System uses **IBM Granite 3** and **RAG Technology**.
            
            **Features:**
            - ΓÜí Instant Summaries
            - ≡ƒöì Source Citations
            - ≡ƒöÉ Enterprise Security
            """)
            st.markdown("---")
            st.caption("Powered by LangChain & WatsonX")

        # RIGHT: Login Form
        with col2:
            with st.container():
                st.markdown("### Get Started")
                tab1, tab2, tab3 = st.tabs(["Log In", "Sign Up", "Recover"])

                # TAB 1: LOGIN
                with tab1:
                    username = st.text_input("Username", key="login_user")
                    password = st.text_input("Password", type="password", key="login_pass")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Sign In"):
                        if check_login(username, password):
                            st.session_state.logged_in = True
                            st.session_state.current_user = username
                            st.toast("Welcome back!", icon="👋")
                            st.rerun()
                        else:
                            st.error("❌ Incorrect username or password")

                # TAB 2: SIGN UP
                with tab2:
                    if "signup_stage" not in st.session_state: st.session_state.signup_stage = 1
                    
                    if st.session_state.signup_stage == 1:
                        new_user = st.text_input("Username", key="new_user")
                        new_email = st.text_input("Email", key="new_email")
                        new_pass = st.text_input("Password", type="password", key="new_pass", help="12+ chars, mixed case, symbols.")
                        confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
                        
                        if st.button("Verify Email & Create"):
                            valid_p, msg = is_valid_password(new_pass)
                            if not new_user: st.warning("Username missing")
                            elif not is_valid_email(new_email): st.warning("Invalid Email")
                            elif not valid_p: st.warning(msg)
                            elif new_pass != confirm_pass: st.error("Passwords don't match")
                            elif check_email_exists(new_email): st.error("Email already registered!")
                            else:
                                # SEND EMAIL
                                code = ''.join(random.choices(string.digits, k=6))
                                success, email_msg = send_verification_email(new_email, code)
                                if success:
                                    st.session_state.signup_otp = code
                                    st.session_state.signup_data = (new_user, new_email, new_pass)
                                    st.session_state.signup_stage = 2
                                    st.toast("≡ƒôº Check your inbox!", icon="Γ£à")
                                    st.rerun()
                                else:
                                    st.error(email_msg)

                    elif st.session_state.signup_stage == 2:
                        st.info(f"≡ƒôº Code sent to {st.session_state.signup_data[1]}")
                        otp_input = st.text_input("Enter Email Code", key="signup_otp_code")
                        
                        if st.button("Confirm & Register"):
                            if otp_input == st.session_state.signup_otp:
                                u, e, p = st.session_state.signup_data
                                success, db_msg = create_user(u, e, p)
                                if success:
                                    st.success(db_msg)
                                    st.session_state.signup_stage = 1
                                else: st.error(db_msg)
                            else: st.error("Γ¥î Invalid Code")
                        
                        if st.button("Back"):
                            st.session_state.signup_stage = 1
                            st.rerun()

                # TAB 3: RECOVER
                with tab3:
                    if "reset_stage" not in st.session_state: st.session_state.reset_stage = 1
                    
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
                                    st.toast("≡ƒôº Code sent!", icon="Γ£à")
                                    st.rerun()
                                else: st.error(email_msg)
                            else: st.error("Email not found.")
                    
                    elif st.session_state.reset_stage == 2:
                        st.info(f"Check {st.session_state.reset_email} for code")
                        otp_input = st.text_input("Enter Code", key="reset_otp_input")
                        new_p = st.text_input("New Password", type="password", key="reset_new_p")
                        
                        if st.button("Change Password"):
                            if otp_input == st.session_state.reset_otp:
                                valid, msg = is_valid_password(new_p)
                                if valid:
                                    if reset_password_in_db(st.session_state.reset_email, new_p):
                                        st.success("Γ£à Password Changed! Please Log In.")
                                        st.session_state.reset_stage = 1
                                    else: st.error("DB Error")
                                else: st.warning(msg)
                            else: st.error("Invalid Code")

# --------------------------------------------------
# Setup
# --------------------------------------------------
st.set_page_config(page_title="Document Q&A Chatbot", page_icon="🤖", layout="wide")
load_dotenv()


BASE_PROJECT_DIR = "./projects"
GLOBAL_DIR = "./global"
os.makedirs(BASE_PROJECT_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

GLOBAL_CHATS_FILE = os.path.join(GLOBAL_DIR, "chats.json")


# --------------------------------------------------
# JSON helpers
# --------------------------------------------------
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


# --------------------------------------------------
# Projects + chats storage
# --------------------------------------------------
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
    """Basic sanitization to avoid weird paths."""
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
        shutil.move(old_base, new_base)  # moves folder + chroma + chats + input
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


# --------------------------------------------------
# Cache heavy objects (fix long loading)
# --------------------------------------------------
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


def build_chain(persist_dir: str):
    vectorstore = get_vectorstore(persist_dir)
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )


# --------------------------------------------------
# Session state
# --------------------------------------------------
if "current_project" not in st.session_state:
    st.session_state.current_project = "Default"

# active_scope: "global" or "project"
if "active_scope" not in st.session_state:
    st.session_state.active_scope = "global"

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_signup_page()
    st.stop()


# --------------------------------------------------
# Load data
# --------------------------------------------------
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


# --------------------------------------------------
# Sidebar (ChatGPT-like separation)
# --------------------------------------------------
with st.sidebar:
    # Top: New chat (GLOBAL like ChatGPT)
    if st.button("✏️ New chat", use_container_width=True):
        new_chat = make_new_chat()
        global_chats.append(new_chat)
        st.session_state.active_scope = "global"
        st.session_state.active_chat_id = new_chat["id"]
        save_global_chats(global_data)
        st.rerun()

    st.text_input("Search chats", key="search_query", placeholder="Search chats…")
    st.markdown("---")

    # Projects
    st.markdown("### Projects")

    selected_project = st.selectbox(
        "Select project",
        projects,
        index=projects.index(project_name),
    )
    if selected_project != project_name:
        st.session_state.current_project = selected_project
        # do NOT force scope change; keep what user was using
        st.session_state.active_chat_id = None
        st.rerun()

    new_project_name = st.text_input("New project name", placeholder="e.g. Test Confirmation")
    if st.button("New project", use_container_width=True):
        name = _normalize_project_name(new_project_name)
        if name:
            init_project(name)
            st.session_state.current_project = name
            # switch to that project's chats area
            st.session_state.active_scope = "project"
            st.session_state.active_chat_id = None
            st.rerun()

    # ------------------------------
    # Manage selected project (Update + Delete)
    # ------------------------------
    st.markdown("#### Manage project")

    with st.expander("✏️ Rename project", expanded=False):
        if project_name == "Default":
            st.info("Default project cannot be renamed.")
        else:
            rename_to = st.text_input("Rename to", key="rename_project_to", placeholder="e.g. FYP Chatbot")
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

    with st.expander("🗑️ Delete project", expanded=False):
        if project_name == "Default":
            st.info("Default project cannot be deleted.")
        else:
            st.warning("This will permanently delete the project folder (documents, chats, vector DB).")
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

    # Project chats list + project new chat
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

    # GLOBAL chats section (Your chats)
    st.markdown("### Your chats")
    render_chat_list(global_chats, "global")

    st.markdown("---")

    # Uploaded documents (project-specific, kept)
    st.markdown("### 📎 Uploaded documents")
    uploaded_files = get_uploaded_files_for_project(project_name)
    if uploaded_files:
        for f in uploaded_files:
            st.write(f"• {f}")
    else:
        st.write("No documents uploaded yet.")

    uploaded = st.file_uploader("Upload PDF / Word / Text", type=["pdf", "docx", "txt"])


# --------------------------------------------------
# File processing (PROJECT KB only)
# --------------------------------------------------
if uploaded:
    init_project(project_name)
    paths = project_paths(project_name)

    file_path = os.path.join(paths["input"], uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("Indexing document…"):
        query_util.process_document(file_path, persist_directory=paths["chroma"])

    # force fresh vectorstore load next time
    try:
        get_vectorstore.clear()
    except Exception:
        pass

    st.toast("📄 Document indexed!", icon="✅")
    st.rerun()


# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("🤖 Document Q&A Chatbot")
st.markdown("Ask me anything about your documents!")

if current_chat is None:
    st.info("No chat selected.")
    st.stop()

# Render messages
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your documents…")
if prompt:
    # auto-title like GPT (first message)
    if len(current_chat["messages"]) == 0:
        current_chat["title"] = auto_title_from_prompt(prompt)

    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Choose KB:
    # - project chats use project KB
    # - global chats: NO KB by default (like ChatGPT "general chat")
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            docs = []
            try:
                if st.session_state.active_scope == "project":
                    paths = project_paths(project_name)
                    init_project(project_name)
                    chain = build_chain(paths["chroma"])
                    response = chain({"question": prompt})
                    answer = response.get("answer", "⚠ No answer returned.")
                    docs = response.get("source_documents", [])
                else:
                    llm = get_llm()
                    answer = llm.invoke(prompt)
            except Exception as e:
                answer = "❌ Error generating answer. Check your API keys / Watsonx configuration."
                st.error(str(e))

            st.markdown(answer)

            if st.session_state.active_scope == "project" and docs:
                with st.expander("📚 View Sources"):
                    seen = set()
                    for i, doc in enumerate(docs, 1):
                        file = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page")
                        key = (file, page)
                        if key in seen:
                            continue
                        seen.add(key)

                        st.subheader(f"📄 Source {i}")
                        st.write(f"**File:** {file}")
                        if page not in [None, "N/A"]:
                            st.write(f"**Page:** {page}")
                        st.write((doc.page_content or "")[:1500] + "…")
                        st.markdown("---")

    current_chat["messages"].append({"role": "assistant", "content": answer})

    # Persist
    if st.session_state.active_scope == "project":
        save_project_chats(project_name, project_data)
    else:
        save_global_chats(global_data)

    st.rerun()
