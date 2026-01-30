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
