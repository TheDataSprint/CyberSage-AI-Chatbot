"""Streamlit user interface for CyberSage AI. Run the FastAPI backend separately."""
from __future__ import annotations
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", "")
except Exception:
    API_BASE_URL = ""

API_BASE_URL = str(
    API_BASE_URL or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
).rstrip("/")

st.set_page_config(
    page_title="CyberSage AI | Cybersecurity Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: radial-gradient(circle at 80% 0%, #132844 0, #0b1424 38%, #08111d 75%); }
      h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
      .hero { padding: 1.5rem 1.7rem; border: 1px solid rgba(104,190,255,.22); border-radius: 20px;
              background: linear-gradient(115deg, rgba(20,52,83,.9), rgba(13,29,49,.84)); margin-bottom: 1.2rem; }
      .hero small { color: #89c9ff; letter-spacing: .16em; font-weight: 700; }
      .hero h1 { color: #f4f8ff; margin: .4rem 0; font-size: 2.35rem; }
      .hero p { color: #b9c9dc; margin: 0; }
      .metric-card { padding: 1rem; border: 1px solid rgba(123,177,220,.2); border-radius: 14px; background: rgba(17,37,60,.7); }
      .metric-label { color: #9fb5ca; font-size: .82rem; }
      .metric-value { color: #eaf5ff; font-size: 1.35rem; font-weight: 700; }
      [data-testid="stChatMessage"] { border: 1px solid rgba(123,177,220,.14); border-radius: 14px; }
      section[data-testid="stSidebar"] { background: #0a1727; border-right: 1px solid rgba(123,177,220,.16); }
      .stButton button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(method: str, endpoint: str, **kwargs):
    try:
        response = requests.request(method, f"{API_BASE_URL}{endpoint}", timeout=120, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(str(detail))
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Cannot reach the FastAPI backend. Start it in another terminal with "
            "`uvicorn api.main:app --reload --port 8000`."
        ) from exc


st.markdown(
    """
    <div class="hero">
      <small>DEFENSIVE SECURITY • KNOWLEDGE • AWARENESS</small>
      <h1>🛡️ CyberSage AI</h1>
      <p>Your document-grounded companion for understanding cyber threats and staying safer online.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛡️ CyberSage AI")
    st.caption("Cybersecurity knowledge assistant")
    st.divider()
    try:
        health = api_request("GET", "/health")
        st.success("Backend connected")
        st.caption(f"Indexed chunks: {health.get('indexed_chunks', 0)}")
    except RuntimeError as exc:
        st.error("Backend offline")
        st.caption(str(exc))
    st.divider()
    st.markdown("**Knowledge base**")
    st.caption("Upload trusted cybersecurity PDFs and index them before asking document-based questions.")
    st.markdown("**Safety scope**")
    st.caption("Defensive learning, prevention, detection and incident-response awareness.")

chat_tab, knowledge_tab, learn_tab, quiz_tab = st.tabs(
    ["💬 Ask CyberSage", "📚 Knowledge Base", "🧠 Learn", "✅ Quick Quiz"]
)

with chat_tab:
    st.subheader("Ask a cybersecurity question")
    st.caption("Answers are generated from your indexed PDFs. Sources are shown below each response.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📖 View sources"):
                    for source in message["sources"]:
                        st.write(f"• {source['source']} — page {source['page']}")

    question = st.chat_input("Example: What are common signs of a phishing email?")
    if question:
        prior_history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages[-6:]
        ]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base and preparing an answer..."):
                try:
                    result = api_request(
                        "POST", "/chat", json={"question": question, "history": prior_history}
                    )
                    answer = result["answer"]
                    st.markdown(answer)
                    if result.get("sources"):
                        with st.expander("📖 View sources"):
                            for source in result["sources"]:
                                st.write(f"• {source['source']} — page {source['page']}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": result.get("sources", [])}
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"I could not complete that request: {exc}"}
                    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

with knowledge_tab:
    st.subheader("Manage cybersecurity documents")
    st.write("Upload PDF reference material such as security awareness guides, standards and defensive security documentation.")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True,
        help="PDFs are saved in data/raw and indexed into the local Qdrant database.",
    )
    if st.button("Upload selected PDFs", disabled=not uploaded_files):
        successes, failures = 0, []
        progress = st.progress(0)
        for index, uploaded in enumerate(uploaded_files, start=1):
            try:
                result = api_request(
                    "POST", "/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
                st.success(f"Uploaded {result['filename']}")
                successes += 1
            except RuntimeError as exc:
                failures.append(f"{uploaded.name}: {exc}")
            progress.progress(index / len(uploaded_files))
        st.info(f"Uploaded {successes} file(s)." if successes else "No files uploaded.")
        for failure in failures:
            st.error(failure)

    if st.button("⚙️ Index all PDFs in the knowledge base", type="primary"):
        with st.spinner("Extracting PDF text, creating chunks and embeddings, and updating Qdrant..."):
            try:
                result = api_request("POST", "/documents/index")
                st.success(
                    f"Indexing complete: {result['files_seen']} PDF(s), "
                    f"{result['pages_with_text']} text pages, {result['chunks_indexed']} chunks."
                )
                for error in result.get("errors", []):
                    st.warning(error)
            except RuntimeError as exc:
                st.error(str(exc))

    try:
        docs = api_request("GET", "/documents")
        left, right = st.columns(2)
        left.metric("PDF files", len(docs.get("pdf_files", [])))
        right.metric("Indexed chunks", docs.get("indexed_chunks", 0))
        if docs.get("pdf_files"):
            st.markdown("**Uploaded documents**")
            for filename in docs["pdf_files"]:
                st.write(f"📄 {filename}")
        else:
            st.info("No PDFs uploaded yet. Add at least one trusted cybersecurity PDF to begin.")
    except RuntimeError as exc:
        st.warning(str(exc))

with learn_tab:
    st.subheader("Cybersecurity learning areas")
    areas = [
        ("Phishing awareness", "Recognize suspicious messages, spoofed senders, urgent requests and unsafe links."),
        ("Password safety", "Understand unique passwords, password managers and multi-factor authentication."),
        ("Device security", "Learn about updates, screen locks, backups and safe software installation."),
        ("Network safety", "Understand public Wi-Fi risks, secure home routers and safe browsing."),
        ("Incident response", "Learn how to report suspicious activity, preserve evidence and follow response procedures."),
        ("Web application security", "Explore common web security weaknesses from a prevention and secure-development perspective."),
    ]
    for title, description in areas:
        with st.expander(title):
            st.write(description)
            st.caption("Ask CyberSage a question about this area after adding relevant documents to your knowledge base.")

with quiz_tab:
    st.subheader("Phishing awareness quick quiz")
    st.write("You receive an unexpected email claiming your account will be closed unless you click a link immediately. What is the safest first response?")
    choice = st.radio(
        "Choose one answer:",
        ["Click the link quickly to protect the account.", "Reply with your password to verify ownership.", "Avoid the link and verify the message through the organization's official channel."],
        index=None,
    )
    if st.button("Check answer"):
        if choice is None:
            st.warning("Choose an answer first.")
        elif choice.startswith("Avoid the link"):
            st.success("Correct. Verify unexpected requests using a known, official channel rather than the email link.")
        else:
            st.error("Not quite. Do not use links or contact details supplied in a suspicious message. Verify independently.")

st.caption("CyberSage AI is an educational assistant. Follow your organization’s security policies and contact its security team during incidents.")
