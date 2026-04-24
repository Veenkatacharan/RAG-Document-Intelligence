import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📄",
    layout="wide"
)

# ── Session state ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Helper functions ───────────────────────────────────────────────

def check_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        return None


def upload_pdf(file) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def ingest_documents() -> dict:
    try:
        response = requests.post(f"{API_URL}/ingest", timeout=120)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def ask_question(question: str, top_k: int = 5) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Document Manager")

    # health check
    health = check_health()
    if health:
        chunks = health.get("chunks_in_database", 0)
        st.success("API connected")
        st.metric("Chunks in database", chunks)
    else:
        st.error("Cannot connect to API.")
        st.code("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
        st.stop()

    st.divider()

    # how it works
    st.subheader("How it works")
    st.markdown(
        "**1.** Upload a PDF below  \n"
        "**2.** Click **Process Documents**  \n"
        "**3.** Ask questions in the chat  \n\n"
        "Answers are grounded in your documents with source citations."
    )

    st.divider()

    # upload
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload PDF", type="primary", use_container_width=True):
            with st.spinner("Uploading..."):
                result = upload_pdf(uploaded_file)
            if "error" in result:
                st.error(f"Upload failed: {result['error']}")
            else:
                st.success(f"Uploaded: {uploaded_file.name}")
                st.info("Now click **Process Documents** to index it.")

    st.divider()

    # ingest
    st.subheader("Process Documents")
    st.caption("Run after uploading to make documents searchable.")
    if st.button("⚙️ Process Documents", use_container_width=True):
        with st.spinner("Indexing — this may take a minute..."):
            result = ingest_documents()
        if "error" in result:
            st.error(f"Failed: {result['error']}")
        else:
            st.success(result.get("message", "Done"))
            st.metric("Chunks created", result.get("chunks_created", 0))
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ──────────────────────────────────────────────────────

st.title("RAG Document Intelligence")
st.caption(
    "Ask plain English questions about your uploaded documents. "
    "Every answer is grounded in the source text — no hallucination."
)

st.divider()

# empty state — shown when no conversation yet
if not st.session_state.messages:
    st.markdown("### Get started")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**What this system does**")
        st.markdown(
            "- Reads and indexes your PDF documents  \n"
            "- Finds the most relevant passages for your question  \n"
            "- Uses a local LLM to generate a cited answer  \n"
            "- Shows you exactly which page the answer came from"
        )

    with col2:
        st.markdown("**Example questions to try**")
        example_questions = [
            "What is the candidate's educational background?",
            "What technical skills does the candidate have?",
            "What work experience does the candidate have?",
            "What projects are listed under Machine Learning?",
        ]
        for q in example_questions:
            if st.button(q, use_container_width=True, key=q):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

    st.divider()

# chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])} passages retrieved)"):
                for source in message["sources"]:
                    sim = source.get("similarity")
                    sim_str = f" — similarity: `{sim:.3f}`" if sim is not None else ""
                    st.caption(f"📄 **{source['source']}** — page {source['page']}{sim_str}")

# answer any pending user message that was added via example button
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and (
    len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "assistant"
):
    last_question = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating answer..."):
            result = ask_question(last_question)

        if "error" in result:
            answer = f"Error: {result['error']}"
            sources = []
        else:
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])

        st.markdown(answer)

        if sources:
            with st.expander(f"📚 Sources ({len(sources)} passages retrieved)"):
                for source in sources:
                    sim = source.get("similarity")
                    sim_str = f" — similarity: `{sim:.3f}`" if sim is not None else ""
                    st.caption(f"📄 **{source['source']}** — page {source['page']}{sim_str}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating answer..."):
            result = ask_question(prompt)

        if "error" in result:
            answer = f"Error: {result['error']}"
            sources = []
        else:
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])

        st.markdown(answer)

        if sources:
            with st.expander(f"📚 Sources ({len(sources)} passages retrieved)"):
                for source in sources:
                    sim = source.get("similarity")
                    sim_str = f" — similarity: `{sim:.3f}`" if sim is not None else ""
                    st.caption(f"📄 **{source['source']}** — page {source['page']}{sim_str}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
