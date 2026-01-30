import streamlit as st
import query_util
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ibm import WatsonxLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import sqlite3
import os
from dotenv import load_dotenv
import csv
import datetime
import hashlib
import re
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

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
        st.error("⚠️ Could not save feedback! Please close 'feedback.csv'.")

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
    # Simple regex for email validation
    return re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email)

def is_valid_password(password):
    if len(password) < 12:
        return False, "⚠️ Password must be at least 12 characters."
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&_])[A-Za-z\d@$!%*?&_]{12,}$"
    if not re.match(pattern, password):
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
        msg['Subject'] = "🔐 Your Verification Code"

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
        return True, "✅ Account created! Please Login."
    except sqlite3.IntegrityError:
        return False, "⚠️ Username or Email already exists."
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
            return False, "❌ Incorrect Current Password. Changes NOT saved."

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
        return True, "✅ Profile updated! Please re-login."
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
            - ⚡ Instant Summaries
            - 🔍 Source Citations
            - 🔐 Enterprise Security
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
                            st.toast(f"Welcome back!", icon="👋")
                            st.rerun()
                        else:
                            st.error("Invalid credentials")

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
                                    st.toast("📧 Check your inbox!", icon="✅")
                                    st.rerun()
                                else:
                                    st.error(email_msg)

                    elif st.session_state.signup_stage == 2:
                        st.info(f"📧 Code sent to {st.session_state.signup_data[1]}")
                        otp_input = st.text_input("Enter Email Code", key="signup_otp_code")
                        
                        if st.button("Confirm & Register"):
                            if otp_input == st.session_state.signup_otp:
                                u, e, p = st.session_state.signup_data
                                success, db_msg = create_user(u, e, p)
                                if success:
                                    st.success(db_msg)
                                    st.session_state.signup_stage = 1
                                else: st.error(db_msg)
                            else: st.error("❌ Invalid Code")
                        
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
                                    st.toast("📧 Code sent!", icon="✅")
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
                                        st.success("✅ Password Changed! Please Log In.")
                                        st.session_state.reset_stage = 1
                                    else: st.error("DB Error")
                                else: st.warning(msg)
                            else: st.error("Invalid Code")

# --- MAIN APP LOGIC ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_signup_page()
else:
    # --- LOGGED IN VIEW ---
    # --- SIDEBAR: USER DASHBOARD ---
    with st.sidebar:
        # 1. User Header with Avatar
        st.markdown(f"""
        <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin:0; color: #2563eb; font-size: 20px;">👤 {st.session_state.current_user}</h3>
            <p style="margin:0; font-size: 12px; color: #666;">Active Session</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Main Navigation / Settings
        with st.expander("🛠️ Profile Settings", expanded=False):
            
            # A. Update Details Section
            st.caption("📝 Update Details")
            new_user_input = st.text_input("Username", value=st.session_state.current_user)
            
            # Fetch email safely
            current_email_db = ""
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("SELECT email FROM users WHERE username = ?", (st.session_state.current_user,))
                res = c.fetchone()
                if res: current_email_db = res[0]
                conn.close()
            except: pass
            
            new_email_input = st.text_input("Email", value=current_email_db)
            
            st.markdown("---")
            st.caption("Change Password")
            
            # NEW: Old Password Field
            old_pass_input = st.text_input("Current Password (Required)", type="password", key="old_pass_verify")
            new_pass_input = st.text_input("New Password (Optional)", type="password", placeholder="Leave empty to keep same", help="12+ chars, mixed case, symbols.")
            
            if st.button("💾 Save Changes", use_container_width=True):
                # We now pass 'old_pass_input' to the function!
                if not old_pass_input:
                    st.error("⚠️ You must enter your Current Password to save changes.")
                else:
                    success, msg = update_user_details(
                        st.session_state.current_user, 
                        old_pass_input,  # <--- Passed here
                        new_user_input, 
                        new_email_input, 
                        new_pass_input if new_pass_input else None
                    )
                    
                    if success:
                        st.success(msg)
                        # Force logout to refresh session
                        st.session_state.logged_in = False
                        st.session_state.current_user = None
                        st.rerun()
                    else:
                        st.error(msg)
            
            st.markdown("---")
            
            # B. Account Actions Section (Renamed)
            st.caption("⚠️ Account Deletion")
            st.markdown("<div style='font-size: 12px; color: #666; margin-bottom: 10px;'>Permanently remove your account and data.</div>", unsafe_allow_html=True)
            
            if st.button("🗑️ Delete Account", type="primary", use_container_width=True):
                st.session_state.confirm_delete = True
            
            # Confirmation Logic
            if "confirm_delete" in st.session_state and st.session_state.confirm_delete:
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
        
        # 3. Document Uploader
        st.header("📂 Document")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        
        # Spacer
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        
        # 4. Logout Button
        if st.button("🚪 Log Out", type="secondary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.messages = []
            if "qa_chain" in st.session_state: del st.session_state["qa_chain"]
            st.rerun()
            

    # Initialize Logic
    st.title("🤖 Document Q&A Chatbot")
    if "messages" not in st.session_state: st.session_state.messages = []
    if "feedback_history" not in st.session_state: st.session_state.feedback_history = set()

    # Load AI (Same as before)
    if "qa_chain" not in st.session_state:
        with st.spinner("Starting AI..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            llm = WatsonxLLM(
                url="https://us-south.ml.cloud.ibm.com",
                apikey=os.getenv("WATSONX_APIKEY"),
                project_id=os.getenv("IBM_PROJECT_ID"),
                model_id="ibm/granite-3-8b-instruct",
                params={"temperature": 0.1, "max_new_tokens": 512}
            )
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), memory=memory, return_source_documents=True
            )

    # Process Upload
    if uploaded_file:
        file_path = f"./input/{uploaded_file.name}"
        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        query_util.process_pdf(file_path)
        st.success("PDF Processed!")

    # Chat Display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for s in message["sources"]: st.write(f"- Page {s['page']}: {s['content'][:100]}...")

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain({"question": prompt})
                answer = response["answer"]
                sources = [{"page": doc.metadata.get("page", "?"), "content": doc.page_content} for doc in response["source_documents"]]
                
                st.markdown(answer)
                with st.expander("📚 Sources"):
                    for s in sources: st.write(f"- Page {s['page']}: {s['content']}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                st.rerun()