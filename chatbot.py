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
import gc
import time
import subprocess
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
from langchain.prompts import PromptTemplate

# ==================================================
# Config (ONLY ONCE)
# ==================================================
load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"
st.set_page_config(page_title="Document Q&A Chatbot", page_icon="🤖", layout="wide")

# ==================================================
# Follow-up question generation (Wahyu SMART + VARIED)
# ==================================================
STOP_WORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "with", "and", "or",
    "according", "document", "documents", "context", "answer", "question",
    "information", "section", "page", "chapter", "introduction", "conclusion",
    "this", "that", "these", "those", "it", "they", "we", "you", "your",
    "what", "which", "when", "where", "who", "why", "how", "about",
    "define", "definition", "meaning"
}

WORLD_KNOWLEDGE_HINTS = {
    "ceo", "capital", "president", "prime minister", "population", "currency",
    "google", "australia", "microsoft", "facebook", "usa", "uk", "china",
    "weather", "today", "latest news", "stock price"
}

def _normalize(s: str) -> str:
    return " ".join((s or "").split()).strip()

def _norm_key(s: str) -> str:
    return _normalize(s).lower()

def _looks_like_world_knowledge(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in WORLD_KNOWLEDGE_HINTS)

def extract_topic_from_question(question: str):
    q = (question or "").strip()
    if not q:
        return None

    ql = q.lower().strip()
    ql = re.sub(r"^(what|how|why|when|where|who)\s+(is|are|does|do|did|can|could|would|should)?\s*", "", ql).strip()
    ql = re.sub(r"^according to (the )?document\s*", "", ql).strip()
    ql = ql.rstrip("?.! ").strip()

    words = [w for w in re.findall(r"\b[a-z]{2,}\b", ql) if w not in STOP_WORDS]
    if not words:
        return None
    return " ".join(words[:3]).strip() or None

def _extract_cues(answer: str):
    a = (answer or "").lower()
    cues = {
        "types": [],
        "has_purpose": any(x in a for x in ["purpose", "aim", "goal", "prepare", "prepares", "preparing"]),
        "has_steps": any(x in a for x in ["step", "steps", "process", "procedure", "first", "second", "then", "next"]),
        "has_compare": any(x in a for x in ["difference", "differs", "compared", "versus", "vs"]),
        "has_examples": "example" in a or "for instance" in a,
        "has_definition": any(x in a for x in ["is defined as", "refers to", "means", "definition"]),
        "mentions_components": any(x in a for x in ["knowledge", "skills", "values", "beliefs", "habits"]),
    }
    if "formal" in a: cues["types"].append("formal schooling")
    if "informal" in a: cues["types"].append("informal learning")
    if "self-directed" in a or "self directed" in a: cues["types"].append("self-directed study")
    if "experiential" in a: cues["types"].append("experiential learning")
    return cues

def generate_followup_questions(answer: str, original_question: str, conversation_history=None):
    asked = set()
    if conversation_history:
        for msg in conversation_history:
            if msg.get("role") == "assistant":
                prev = msg.get("followups") or msg.get("followup_questions") or []
                for q in prev:
                    asked.add(_norm_key(q))

    def add(q: str, out: list):
        if not q:
            return
        k = _norm_key(q)
        if k in asked:
            return
        if any(_norm_key(x) == k for x in out):
            return
        out.append(_normalize(q))

    topic = extract_topic_from_question(original_question) or "this topic"
    cues = _extract_cues(answer)

    out = []

    if _looks_like_world_knowledge(original_question):
        add("That seems outside the uploaded document. What part of the document should I use?", out)
        add("What topics are covered in the uploaded document?", out)
        add("Can you ask a question related to the document content?", out)
        return out[:3]

    if any(x in (answer or "").lower() for x in [
        "cannot find", "not in the document", "not found in the document", "i can’t find", "i can't find"
    ]):
        add(f"Where in the document is {topic} discussed?", out)
        add(f"What does the document say about {topic} overall?", out)
        add("What related topics are covered in the document?", out)
        return out[:3]

    candidate_pool = []

    if cues["has_definition"]:
        candidate_pool.extend([
            f"What is the document’s definition of {topic}?",
            f"What key points are included in the definition of {topic}?",
        ])

    if cues["has_purpose"]:
        candidate_pool.extend([
            f"What are the main goals of {topic}?",
            f"Why is {topic} important according to the document?",
        ])
    else:
        candidate_pool.append(f"What is the purpose of {topic} in this context?")

    if cues["types"]:
        candidate_pool.append(f"What types of {topic} are mentioned in the document?")
        if len(cues["types"]) >= 2:
            t1, t2 = cues["types"][0], cues["types"][1]
            candidate_pool.append(f"How does {t1} differ from {t2}?")

    if cues["has_steps"]:
        candidate_pool.extend([
            f"What are the steps or process related to {topic}?",
            f"What happens after the first step in {topic}?",
        ])

    candidate_pool.append(f"Can you give an example of {topic} from the document?")

    if cues["mentions_components"]:
        candidate_pool.append(f"What key components of {topic} are described (e.g., knowledge/skills/values)?")

    candidate_pool.extend([
        f"What is the key takeaway about {topic}?",
        f"What are the limitations or challenges of {topic} mentioned?",
    ])

    random.shuffle(candidate_pool)

    for q in candidate_pool:
        add(q, out)
        if len(out) == 3:
            break

    if len(out) < 3:
        add(f"What does the document say about {topic}?", out)
    if len(out) < 3:
        add("Can you point me to the relevant section in the document?", out)

    return out[:3]

# ==================================================
# Feedback (Kaixin/Xinru) — persist per message
# ==================================================
def save_feedback(query, response, rating, comment=""):
    try:
        file_exists = os.path.isfile("feedback.csv")
        with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Query", "Response", "Rating", "Comment"])
            writer.writerow([datetime.datetime.now(), query, response, rating, comment])
    except PermissionError:
        st.error("⚠️ Could not save feedback! Please close 'feedback.csv'.")

# ==================================================
# Login + Email verification + Profile Settings (Kaixin)
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
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = "🔐 Your Verification Code"

        body = f"""
        <html><body>
            <h2>Verification Code</h2>
            <p>Your secure code is:</p>
            <h1 style="color: #2563eb;">{code}</h1>
        </body></html>
        """
        msg.attach(MIMEText(body, "html"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        return True, "✅ Email Sent! Check your inbox."
    except Exception as e:
        return False, f"❌ Email Failed: {str(e)}"

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password_hash TEXT
        )
    """)
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# ==================================================
# Per-user storage roots (Kaixin)
# ==================================================
BASE_PROJECT_DIR = "./projects"
GLOBAL_DIR = "./global"
os.makedirs(BASE_PROJECT_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

def get_user_projects_root(username: str):
    root = os.path.join(BASE_PROJECT_DIR, username)
    os.makedirs(root, exist_ok=True)
    return root

def get_user_global_dir(username: str):
    root = os.path.join(GLOBAL_DIR, username)
    os.makedirs(root, exist_ok=True)
    return root

def create_user(username, email, password):
    """Creates a user. Cleans any old locked folders to prevent WinError issues."""
    user_proj_dir = os.path.join(BASE_PROJECT_DIR, username)
    user_global_dir = os.path.join(GLOBAL_DIR, username)

    for folder in [user_proj_dir, user_global_dir]:
        if os.path.exists(folder):
            try:
                if os.path.isdir(folder): shutil.rmtree(folder)
                else: os.remove(folder)
            except OSError:
                try:
                    subprocess.run(f'rmdir /S /Q "{folder}"', shell=True, check=False)
                except Exception:
                    try:
                        new_name = f"{folder}_junk_{uuid.uuid4()}"
                        os.rename(folder, new_name)
                    except Exception:
                        return False, "❌ FATAL ERROR: Please CLOSE VS Code and try again."

    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, email, make_hashes(password)))
        conn.commit()
        conn.close()
        return True, "✅ Account created! Please Login."
    except sqlite3.IntegrityError:
        return False, "⚠️ Username or Email already exists."
    except Exception as e:
        return False, f"Error: {e}"

def delete_user_account(username):
    """Deletes user from DB and attempts to delete/rename their data folders."""
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        conn.close()

        st.cache_resource.clear()
        gc.collect()
        time.sleep(0.3)

        for root_dir in [BASE_PROJECT_DIR, GLOBAL_DIR]:
            user_dir = os.path.join(root_dir, username)
            if os.path.exists(user_dir):
                try:
                    if os.path.isdir(user_dir): shutil.rmtree(user_dir)
                    else: os.remove(user_dir)
                except OSError:
                    try:
                        trash_name = f"{user_dir}_trash_{uuid.uuid4()}"
                        os.rename(user_dir, trash_name)
                    except OSError:
                        pass

        return True
    except Exception:
        return False

def check_login(username, password):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        data = c.fetchone()
        conn.close()
        return bool(data and check_hashes(password, data[0]))
    except Exception:
        return False

def check_email_exists(email):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE email = ?", (email,))
        data = c.fetchone()
        conn.close()
        return data[0] if data else None
    except Exception:
        return None

def reset_password_in_db(email, new_password):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("UPDATE users SET password_hash = ? WHERE email = ?", (make_hashes(new_password), email))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def update_user_details(current_username, old_password, new_username, new_email, new_password=None):
    """Updates profile. Copies per-user data folders safely on Windows."""
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()

        c.execute("SELECT password_hash, email FROM users WHERE username = ?", (current_username,))
        row = c.fetchone()
        if not row:
            conn.close()
            return False, "User not found."

        stored_hash, current_email = row
        if not check_hashes(old_password, stored_hash):
            conn.close()
            return False, "❌ Incorrect Current Password. Changes NOT saved."

        if new_username != current_username:
            c.execute("SELECT 1 FROM users WHERE username = ?", (new_username,))
            if c.fetchone():
                conn.close()
                return False, "Username already taken."

        if new_email and new_email != current_email:
            c.execute("SELECT 1 FROM users WHERE email = ?", (new_email,))
            if c.fetchone():
                conn.close()
                return False, "Email already in use."

        warning_msg = ""
        if new_username != current_username:
            st.cache_resource.clear()
            gc.collect()
            time.sleep(0.2)

            old_proj = os.path.join(BASE_PROJECT_DIR, current_username)
            new_proj = os.path.join(BASE_PROJECT_DIR, new_username)
            old_glob = os.path.join(GLOBAL_DIR, current_username)
            new_glob = os.path.join(GLOBAL_DIR, new_username)

            try:
                if os.path.exists(new_proj):
                    shutil.rmtree(new_proj, ignore_errors=True)
                if os.path.exists(new_glob):
                    shutil.rmtree(new_glob, ignore_errors=True)

                if os.path.exists(old_proj):
                    shutil.copytree(old_proj, new_proj)
                if os.path.exists(old_glob):
                    shutil.copytree(old_glob, new_glob)

                try:
                    if os.path.exists(old_proj):
                        shutil.rmtree(old_proj, ignore_errors=True)
                    if os.path.exists(old_glob):
                        shutil.rmtree(old_glob, ignore_errors=True)
                except OSError:
                    warning_msg = f" (Note: Old folder '{current_username}' is locked. Delete manually later.)"

            except Exception as e:
                conn.close()
                return False, f"❌ Data Migration Failed: {e}"

        if new_password:
            c.execute(
                "UPDATE users SET username = ?, email = ?, password_hash = ? WHERE username = ?",
                (new_username, new_email, make_hashes(new_password), current_username),
            )
        else:
            c.execute(
                "UPDATE users SET username = ?, email = ? WHERE username = ?",
                (new_username, new_email, current_username),
            )

        conn.commit()
        conn.close()
        return True, f"✅ Profile updated! Please re-login.{warning_msg}"
    except Exception as e:
        return False, f"Error: {e}"

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
                    code = "".join(random.choices(string.digits, k=6))
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
                    code = "".join(random.choices(string.digits, k=6))
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
# Projects + Chats (Your base, upgraded to per-user paths)
# ==================================================
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
    user_root = get_user_projects_root(st.session_state.current_user)
    base = os.path.join(user_root, project_name)
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
        return False, "Invalid names."
    if old_name == "Default":
        return False, "Cannot rename Default."
    if old_name == new_name:
        return False, "New name is the same."

    old_base = project_paths(old_name)["base"]
    new_base = project_paths(new_name)["base"]

    if not os.path.exists(old_base):
        return False, "Project not found."
    if os.path.exists(new_base):
        return False, "Target exists."

    # Safe copy strategy (Windows)
    try:
        shutil.copytree(old_base, new_base)
    except Exception as e:
        return False, f"Copy failed: {e}"

    deletion_success = False
    try:
        shutil.rmtree(old_base)
        deletion_success = True
    except Exception:
        pass

    if not deletion_success:
        try:
            subprocess.run(f'rmdir /S /Q "{old_base}"', shell=True, check=False)
            deletion_success = True
        except Exception:
            pass

    if not deletion_success:
        try:
            trash_name = f"{old_base}_trash_{uuid.uuid4()}"
            os.rename(old_base, trash_name)
            deletion_success = True
        except Exception:
            pass

    if not deletion_success:
        try:
            with open(os.path.join(old_base, ".deleted"), "w") as f:
                f.write("deleted")
        except Exception:
            pass

    init_project(new_name)
    return True, f"✅ Project renamed to '{new_name}'."

def delete_project(name: str):
    name = _normalize_project_name(name)
    if name == "Default":
        return False, "Cannot delete Default."

    base = project_paths(name)["base"]
    if not os.path.exists(base):
        return False, "Project not found."

    st.cache_resource.clear()
    gc.collect()
    time.sleep(0.2)

    try:
        shutil.rmtree(base)
        return True, "✅ Project deleted."
    except Exception:
        pass

    try:
        trash = f"{base}_deleted_{uuid.uuid4()}"
        os.rename(base, trash)
        return True, "✅ Project deleted."
    except Exception:
        pass

    try:
        with open(os.path.join(base, ".deleted"), "w") as f:
            f.write("This project is deleted.")
        return True, "✅ Project deleted."
    except Exception as e:
        return False, f"❌ Could not delete or mark project: {e}"

def list_projects():
    user_root = get_user_projects_root(st.session_state.current_user)
    os.makedirs(user_root, exist_ok=True)

    init_project("Default")
    projs = ["Default"]

    for d in os.listdir(user_root):
        if "_trash_" in d or "_zombie_" in d or "_deleted_" in d:
            continue
        p = os.path.join(user_root, d)
        if os.path.isdir(p) and d != "Default":
            if not os.path.exists(os.path.join(p, ".deleted")):
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
    user_dir = get_user_global_dir(st.session_state.current_user)
    path = os.path.join(user_dir, "chats.json")
    if not os.path.exists(path):
        _safe_save_json(path, {"chats": []})
    data = _safe_load_json(path, {"chats": []})
    if "chats" not in data or not isinstance(data["chats"], list):
        data = {"chats": []}
    return data

def save_global_chats(data):
    user_dir = get_user_global_dir(st.session_state.current_user)
    path = os.path.join(user_dir, "chats.json")
    _safe_save_json(path, data)

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
# LangChain / RAG (Wahyu accuracy upgrades + consistent embeddings)
# ==================================================
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

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

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5}
    )

    prompt_template = """
You are a precise document assistant.

Answer using ONLY the context below.
If the answer is not in the context, say exactly:
"I cannot find this information in the document."

Do NOT repeat the question.
Return ONLY the final answer.

Context:
{context}

Question:
{question}

Answer:
"""
    QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

def safe_process_document(path: str, persist_dir: str):
    # Always pass persist_directory correctly
    if hasattr(query_util, "process_document"):
        query_util.process_document(path, persist_directory=persist_dir)
        return
    if hasattr(query_util, "process_pdf"):
        query_util.process_pdf(path, persist_directory=persist_dir)
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
    st.session_state.active_scope = "project"  # default to project now
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None
if "chat_memories" not in st.session_state:
    st.session_state.chat_memories = {}

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
# Sidebar (projects + chats + upload + profile)
# ==================================================
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state.current_user}")

    # Profile settings (Kaixin)
    with st.expander("🛠️ Profile Settings", expanded=False):
        st.caption("📝 Update Details")
        new_user_input = st.text_input("Username", value=st.session_state.current_user)

        current_email_db = ""
        try:
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT email FROM users WHERE username = ?", (st.session_state.current_user,))
            res = c.fetchone()
            if res:
                current_email_db = res[0]
            conn.close()
        except Exception:
            pass

        new_email_input = st.text_input("Email", value=current_email_db)

        st.markdown("---")
        st.caption("🔐 Security Check")
        old_pass_input = st.text_input("Current Password (Required)", type="password", key="old_pass_verify")
        new_pass_input = st.text_input("New Password (Optional)", type="password", placeholder="Leave empty to keep same")

        if st.button("💾 Save Changes", use_container_width=True):
            if not old_pass_input:
                st.error("⚠️ You must enter your Current Password to save changes.")
            else:
                success, msg = update_user_details(
                    st.session_state.current_user,
                    old_pass_input,
                    new_user_input,
                    new_email_input,
                    new_pass_input if new_pass_input else None
                )
                if success:
                    st.success(msg)
                    st.session_state.logged_in = False
                    st.session_state.current_user = None
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("---")
        st.caption("⚠️ Account Deletion")
        st.markdown("<div style='font-size: 12px; color: #666; margin-bottom: 10px;'>Permanently remove your account and data.</div>", unsafe_allow_html=True)

        if st.button("🗑️ Delete Account", type="primary", use_container_width=True):
            st.session_state.confirm_delete = True

        if st.session_state.get("confirm_delete", False):
            st.warning("Are you sure?", icon="⚠️")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes", use_container_width=True):
                    if delete_user_account(st.session_state.current_user):
                        st.success("Deleted.")
                        st.session_state.logged_in = False
                        st.session_state.current_user = None
                        st.rerun()
            with col_no:
                if st.button("No", use_container_width=True):
                    st.session_state.confirm_delete = False
                    st.rerun()

    st.markdown("---")

    if st.button("✏️ New global chat", use_container_width=True):
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
        st.session_state.active_scope = "project"
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
                    st.session_state.active_scope = "project"
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
    st.markdown("### Your global chats")
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

    st.markdown("---")
    if st.button("🚪 Log out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

# ==================================================
# File processing (PROJECT KB) + Loop prevention (Kaixin)
# ==================================================
if uploaded:
    init_project(project_name)
    paths = project_paths(project_name)

    file_path = os.path.join(paths["input"], uploaded.name)

    # Prevent re-indexing same file on rerun
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Indexing document..."):
            safe_process_document(file_path, persist_dir=paths["chroma"])

        st.cache_resource.clear()
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
            # Handle both dict (from JSON) and Document objects
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                content = doc.get("page_content", "")
            else:
                metadata = getattr(doc, "metadata", {}) or {}
                content = getattr(doc, "page_content", "") or ""

            file = metadata.get("source", "Unknown")
            page = metadata.get("page", "N/A")
            key = (file, str(page))
            if key in seen:
                continue
            seen.add(key)

            st.subheader(f"📄 Source {idx}")
            st.write(f"**File:** {os.path.basename(file)} | **Page:** {page}")
            st.write(content.strip()[:1500] + ("…" if len(content.strip()) > 1500 else ""))
            st.markdown("---")
            idx += 1

# Render messages + follow-ups + feedback
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

        # Feedback UI for assistant messages (Kaixin)
        if msg["role"] == "assistant" and not msg.get("feedback_submitted", False):
            with st.expander("⭐ Rate this response"):
                feedback_key = f"fb_{current_chat['id']}_{mi}"
                with st.form(key=f"form_{feedback_key}", clear_on_submit=False):
                    rating = st.radio("Rating", ["Positive", "Negative"], horizontal=True)
                    comment = st.text_area("Optional: Why this rating?", placeholder="Your thoughts...")
                    submit_clicked = st.form_submit_button("Submit Feedback", use_container_width=True)

                    if submit_clicked:
                        user_q = current_chat["messages"][mi-1]["content"] if mi > 0 else "N/A"
                        save_feedback(user_q, msg["content"], rating, comment)

                        msg["feedback_submitted"] = True

                        if st.session_state.active_scope == "project":
                            save_project_chats(project_name, project_data)
                        else:
                            save_global_chats(global_data)

                        st.toast("Feedback submitted!", icon="✅")
                        st.rerun()

# Input (follow-up click -> queued_prompt)
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

    docs = []
    answer = ""
    scope = st.session_state.active_scope

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if scope == "project":
                    paths = project_paths(project_name)
                    init_project(project_name)
                    memory = get_memory_for_chat(scope, current_chat["id"])
                    chain = build_chain(paths["chroma"], memory=memory)

                    response = chain.invoke({"question": prompt})
                    answer = response.get("answer", "") or "⚠ No answer returned."
                    docs = response.get("source_documents", []) or []

                else:
                    # Keep global chat (your original) but it won't cite documents
                    llm = get_llm()
                    answer = llm.invoke(prompt)
                    docs = []

            except Exception as e:
                answer = "❌ Error generating answer. Check your API keys / Watsonx configuration."
                st.error(str(e))

            st.markdown(answer)

            if scope == "project" and docs:
                render_sources(docs)

            followups = generate_followup_questions(answer, prompt, conversation_history=current_chat["messages"])

            if followups:
                st.markdown("---")
                st.markdown("**💡 You might also want to ask:**")
                for qi, fq in enumerate(followups):
                    if st.button(fq, key=f"fu_live_{current_chat['id']}_{qi}", use_container_width=True):
                        st.session_state.queued_prompt = fq
                        st.rerun()

    # Serialize docs for safe JSON saving (Kaixin)
    serialized_docs = []
    if scope == "project" and docs:
        for d in docs:
            serialized_docs.append({
                "page_content": getattr(d, "page_content", ""),
                "metadata": getattr(d, "metadata", {}) or {}
            })

    current_chat["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": serialized_docs if scope == "project" else [],
        "followups": followups,
        "feedback_submitted": False
    })

    if scope == "project":
        save_project_chats(project_name, project_data)
    else:
        save_global_chats(global_data)

    st.rerun()
