import os
import io
import re
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

import query_util
import ocr_util

load_dotenv()

st.set_page_config(page_title="Image + Document Q&A Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Image + Document Q&A Chatbot")
st.markdown("Upload an image + select a document. Answers must come from the **selected document only**.")

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "ocr_name" not in st.session_state:
    st.session_state.ocr_name = None


# -----------------------------
# Helpers
# -----------------------------
def list_md_sources(md_folder: str = "output_md"):
    p = Path(md_folder)
    if not p.exists():
        return []
    return sorted([str(x).replace("\\", "/") for x in p.glob("*.md")])


def is_who_question(q: str) -> bool:
    ql = (q or "").strip().lower()
    return ql.startswith("who") or "who is" in ql or "who's" in ql


def extract_name_from_text(text: str) -> str | None:
    """
    Prefer First M. Last (Jennifer G. Beasley)
    Else First Last (Myra Haulmark)
    """
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()

    pat_mid = re.compile(r"\b([A-Z][a-z]+)\s([A-Z]\.)\s([A-Z][a-z]+)\b")
    m = pat_mid.search(cleaned)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}".strip()

    pat_2 = re.compile(r"\b([A-Z][a-z]+)\s([A-Z][a-z]+)\b")
    candidates = pat_2.findall(cleaned)
    if candidates:
        return f"{candidates[0][0]} {candidates[0][1]}".strip()

    return None


def _is_junk_line(line: str) -> bool:
    """Filter out attribution/license stuff that often contains author names."""
    low = line.lower()
    junk_markers = [
        "copyright", "creative commons", "licensed", "license",
        "cc by", "cc-by", "cc by-sa", "cc-by-sa",
        "attribution", "sharealike", "all rights reserved",
        "university of", "libraries", "fayetteville",
        "adapted from", "modified from"
    ]
    return any(m in low for m in junk_markers)


def extract_bio_from_markdown(md_path: str, person_name: str) -> str | None:
    """
    ✅ Stronger bio extraction:
    - Locate "Author Biographies" section first
    - Find the person's name inside that section
    - Extract the bio paragraph(s), skipping attribution/license lines
    """
    if not md_path or not os.path.exists(md_path):
        return None
    name = (person_name or "").strip()
    if not name:
        return None

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    name_low = name.lower()

    # 1) Find start of Author Biographies section
    bio_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("#") and "author biographies" in line.lower():
            bio_start = i
            break

    # If no Author Biographies heading exists, fallback to whole doc scan (but still skip junk)
    scan_start = bio_start if bio_start is not None else 0

    # 2) Determine scan end: next big heading after Author Biographies
    scan_end = len(lines)
    if bio_start is not None:
        for j in range(bio_start + 1, len(lines)):
            if lines[j].strip().startswith("#") and "author biographies" not in lines[j].lower():
                scan_end = j
                break

    # 3) Find the person inside the section (prefer heading or bold line)
    hit = None
    for i in range(scan_start, scan_end):
        low = lines[i].lower()
        if name_low in low:
            # avoid false hits in junk lines
            if _is_junk_line(lines[i]):
                continue
            hit = i
            break

    if hit is None:
        return None

    # 4) Collect bio text after the hit line
    out = []

    # If the hit line is a heading like "## Myra Haulmark, Ed.D."
    # include it only as anchor, but bio is usually below it.
    for k in range(hit + 1, scan_end):
        ln = lines[k].rstrip("\n")

        # stop at next heading or next author-like heading
        if ln.strip().startswith("#"):
            break

        # stop if we reach another author name-like line (common in author lists)
        # (e.g., "Jennifer G. Beasley, Ed.D.")
        if re.search(r"\b[A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-z]+.*Ed\.D\.\b", ln):
            break

        # skip junk lines
        if _is_junk_line(ln):
            continue

        out.append(ln)

    text = "\n".join(out).strip()

    # Remove markdown image refs
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # If we accidentally got something too short, treat as not found
    if len(text) < 60:
        return None

    return text


def shorten_to_intro(text: str, max_chars: int = 700) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(sentences[:3]).strip()
    return (out[:max_chars].rstrip() + "…") if len(out) > max_chars else out


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📌 Source Selection (Important)")

    md_sources = list_md_sources("output_md")
    if md_sources:
        labels = {s: Path(s).name for s in md_sources}
        selected = st.selectbox(
            "Indexed document source",
            options=md_sources,
            format_func=lambda x: labels.get(x, x),
            index=md_sources.index(st.session_state.selected_source)
            if st.session_state.selected_source in md_sources else 0
        )
        st.session_state.selected_source = selected
    else:
        st.info("No markdown sources found in `output_md/` yet.")

    st.markdown("---")

    st.header("🖼 Upload Image")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    st.markdown("---")

    st.header("📄 Upload PDF (optional)")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    st.markdown("---")
    if st.button("(Re)Index selected document"):
        if st.session_state.selected_source:
            with st.spinner("Indexing selected document..."):
                query_util.ensure_indexed(st.session_state.selected_source)
            st.success("Done indexing.")


# -----------------------------
# PDF upload -> convert + index
# -----------------------------
if uploaded_pdf:
    os.makedirs("uploaded_docs", exist_ok=True)
    pdf_path = os.path.join("uploaded_docs", uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    with st.spinner("Processing PDF -> markdown -> indexing..."):
        md_path = query_util.process_pdf(pdf_path)
        st.session_state.selected_source = md_path
        query_util.ensure_indexed(md_path)

    st.success(f"PDF indexed: {uploaded_pdf.name}")


# -----------------------------
# Image handling (OCR)
# -----------------------------
if uploaded_image:
    st.session_state.image_bytes = uploaded_image.getvalue()
    pil_img = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Reading text from the image..."):
        st.session_state.ocr_text = ocr_util.extract_text_from_image(pil_img) or ""
        st.session_state.ocr_name = extract_name_from_text(st.session_state.ocr_text)


# -----------------------------
# Chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask a question about the image + selected document")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            selected_source = st.session_state.selected_source
            if not selected_source:
                answer = "No document selected. Please select a document source from the sidebar."
            else:
                query_util.ensure_indexed(selected_source)

                if is_who_question(user_input) and st.session_state.ocr_name:
                    ocr_name = st.session_state.ocr_name

                    bio = extract_bio_from_markdown(selected_source, ocr_name)

                    if not bio:
                        # fallback: retrieval by name (still document-only)
                        docs = query_util.retrieve_from_source(query=ocr_name, md_path=selected_source, k=8)
                        bio = docs[0].page_content if docs else ""

                    answer = (
                        f"**{ocr_name}**\n\n"
                        f"{shorten_to_intro(bio)}\n\n"
                        f"✅ OCR detected name: `{ocr_name}`\n\n"
                        f"Source: `{selected_source}`"
                    )
                else:
                    # Non-who questions: use question + OCR text as query
                    q = user_input
                    if st.session_state.ocr_text.strip():
                        q = f"Image text:\n{st.session_state.ocr_text}\n\nQuestion:\n{user_input}"

                    docs = query_util.retrieve_from_source(query=q, md_path=selected_source, k=6)
                    if docs:
                        best = re.sub(r"\s+", " ", docs[0].page_content).strip()
                        answer = f"{shorten_to_intro(best)}\n\nSource: `{selected_source}`"
                    else:
                        answer = "Not found in the selected document."

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


