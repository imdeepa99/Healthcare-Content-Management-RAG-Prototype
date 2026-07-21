import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.vector_store import create_vector_store
from src.rag_pipeline import answer_question, compare_papers

load_dotenv()

st.set_page_config(
    page_title="Healthcare Content Management Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
:root {
    --bg-1: #050816;
    --bg-2: #0b1020;
    --bg-3: #111827;
    --card: rgba(15, 23, 42, 0.72);
    --card-2: rgba(17, 24, 39, 0.78);
    --line: rgba(255,255,255,0.08);
    --text: #f8fafc;
    --muted: #94a3b8;
    --blue: #60a5fa;
    --violet: #8b5cf6;
    --cyan: #22d3ee;
    --shadow: 0 10px 30px rgba(0,0,0,0.28);
}

/* App background */
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(96,165,250,0.14), transparent 26%),
        radial-gradient(circle at 82% 18%, rgba(139,92,246,0.12), transparent 26%),
        radial-gradient(circle at 50% 78%, rgba(34,211,238,0.08), transparent 28%),
        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(7,11,24,0.96), rgba(15,23,42,0.96));
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}
.sidebar-brand {
    padding: 0.6rem 0 1rem 0;
}
.brand-badge {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(96,165,250,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 0.82rem;
    color: #dbeafe;
    margin-bottom: 0.75rem;
}
.brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.2rem;
}
.brand-sub {
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.45;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    background:
        linear-gradient(135deg, rgba(15,23,42,0.84), rgba(17,24,39,0.80)),
        radial-gradient(circle at top left, rgba(96,165,250,0.14), transparent 30%);
    box-shadow: var(--shadow);
    padding: 32px 34px;
    margin-bottom: 22px;
}
.hero-shell::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 18% 10%, rgba(96,165,250,0.14), transparent 20%),
        radial-gradient(circle at 78% 14%, rgba(139,92,246,0.15), transparent 20%),
        radial-gradient(circle at 60% 80%, rgba(34,211,238,0.08), transparent 20%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-block;
    font-size: 0.85rem;
    color: #dbeafe;
    padding: 0.36rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    margin-bottom: 0.9rem;
}
.hero-title {
    font-size: 3rem;
    line-height: 1.02;
    font-weight: 900;
    margin-bottom: 0.7rem;
    letter-spacing: -0.03em;
    color: #f8fafc;
}
.hero-title .grad {
    background: linear-gradient(90deg, #93c5fd, #c4b5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    max-width: 920px;
    color: #cbd5e1;
    font-size: 1.03rem;
    line-height: 1.7;
}
.hero-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}
.hero-pill {
    border-radius: 16px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    color: #dbeafe;
    font-size: 0.94rem;
}

.glass-card {
    background: rgba(15,23,42,0.64);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    box-shadow: var(--shadow);
    padding: 20px 22px;
    margin-bottom: 18px;
}
.subtle-card {
    background: rgba(17,24,39,0.72);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.22);
    padding: 16px 18px;
    margin-bottom: 16px;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.35rem;
}
.section-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 0.8rem;
}

.metric-wrap {
    background: linear-gradient(180deg, rgba(17,24,39,0.92), rgba(15,23,42,0.92));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}
.metric-label {
    color: #93a3b8;
    font-size: 0.92rem;
    margin-bottom: 0.45rem;
}
.metric-value {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 900;
    letter-spacing: -0.02em;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 16px;
    padding: 0.78rem 1.1rem;
    font-weight: 800;
    color: white !important;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 55%, #06b6d4 100%);
    box-shadow: 0 10px 24px rgba(37,99,235,0.24);
    transition: all 0.18s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.06);
}
.stDownloadButton > button {
    width: 100%;
    border: none;
    border-radius: 16px;
    padding: 0.78rem 1.1rem;
    font-weight: 800;
    color: white !important;
    background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
    box-shadow: 0 10px 24px rgba(8,145,178,0.22);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    color: #cbd5e1;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.05);
    padding: 11px 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(37,99,235,0.20), rgba(139,92,246,0.20));
    color: white !important;
    border: 1px solid rgba(96,165,250,0.35);
    box-shadow: 0 6px 18px rgba(59,130,246,0.12);
}

.stTextInput > div > div > input,
.stTextArea textarea,
[data-testid="stFileUploader"] section,
[data-testid="stChatInput"] textarea {
    background: rgba(15,23,42,0.9) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    border: 1px dashed rgba(255,255,255,0.13);
    padding: 0.45rem;
}
[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0;
    padding-top: 0.6rem;
    background: linear-gradient(180deg, rgba(5,8,22,0.0), rgba(5,8,22,0.96) 40%);
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 0.8rem;
    box-shadow: 0 8px 18px rgba(0,0,0,0.14);
}
.chat-note {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-bottom: 0.6rem;
}

.streamlit-expanderHeader {
    background: rgba(15,23,42,0.84);
    border-radius: 14px;
}

.stAlert {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.07);
}

.footer-card {
    margin-top: 22px;
    border-radius: 22px;
    background: rgba(17,24,39,0.58);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 16px 18px;
    color: #94a3b8;
    font-size: 0.92rem;
    text-align: center;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 2.2rem;
    }
    .hero-grid {
        grid-template-columns: 1fr;
    }
}
/* ===== Selectbox Dark Theme ===== */

div[data-baseweb="select"] > div {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}

div[data-baseweb="select"] span {
    color: #f8fafc !important;
}

div[data-baseweb="select"] svg {
    fill: #f8fafc !important;
}

/* Dropdown menu */
div[role="listbox"] {
    background: rgba(15,23,42,0.98) !important;
    color: white !important;
    border-radius: 12px !important;
}

div[role="option"] {
    color: white !important;
}

div[role="option"]:hover {
    background: rgba(37,99,235,0.25) !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_STATE = {
    "vector_store": None,
    "documents_processed": False,
    "document_names": [],
    "total_chunks": 0,
    "chat_history": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_workspace():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value


def process_documents(files, chunk_size, chunk_overlap):
    all_chunks = []
    document_names = []

    for uploaded_file in files:
        document_names.append(uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_path = temp_pdf.name

        try:
            pages = extract_text_from_pdf(temp_path, uploaded_file.name)
            chunks = chunk_pages(
                pages=pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return all_chunks, document_names


def show_sources(documents):
    if not documents:
        return

    with st.expander("View supporting evidence"):
        for index, doc in enumerate(documents, start=1):
            name = doc.metadata.get("paper_name", "Unknown document")
            page = doc.metadata.get("page", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "Unknown")
            st.markdown(f"**Source {index}:** {name} | Page {page} | {chunk_id}")
            st.write(doc.page_content)
            st.markdown("---")


def ask_rag(prompt, model_choice, top_k):
    return answer_question(
        st.session_state["vector_store"],
        prompt,
        chat_history=st.session_state["chat_history"],
        per_paper=2,
        total_k=max(top_k, 12),
        model_name=model_choice,
    )


with st.sidebar:
    st.markdown("## 🏥 Healthcare AI Assistant")
    st.write(
        "Upload healthcare policies, coverage documents, provider manuals, "
        "and clinical guidelines."
    )

    st.markdown("### Settings")
    top_k = st.slider("Evidence passages", 2, 15, 8)
    chunk_size = st.slider("Chunk size", 500, 1500, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 200, 50)
    model_choice = st.selectbox("AI model", ["gemini", "groq", "ollama"], index=0)

    st.markdown("---")
    st.markdown("### Example questions")
    st.markdown(
        """
        - What services are covered?
        - Is prior authorization required?
        - List all CPT codes.
        - What are the eligibility criteria?
        - What exclusions are listed?
        - What documentation is required?
        """
    )

st.markdown(
    """
    <div class="hero">
        <h1>Healthcare Content Management Assistant</h1>
        <p>
            Upload healthcare policies, provider manuals, clinical guidelines,
            or coverage documents. Ask questions, create summaries, compare
            policies, and extract important rules with evidence from the source files.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📂 Upload Healthcare Documents</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Upload one or more PDF healthcare documents.</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload healthcare PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} document(s) ready for processing.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Process Documents"):
            with st.spinner("Building the healthcare knowledge base..."):
                try:
                    chunks, names = process_documents(
                        uploaded_files,
                        chunk_size,
                        chunk_overlap,
                    )

                    if not chunks:
                        st.error("No readable text was found in the uploaded PDFs.")
                    else:
                        st.session_state["vector_store"] = create_vector_store(chunks)
                        st.session_state["documents_processed"] = True
                        st.session_state["document_names"] = names
                        st.session_state["total_chunks"] = len(chunks)
                        st.session_state["chat_history"] = []
                        st.success("Documents processed successfully.")
                        st.rerun()
                except Exception as error:
                    st.error(f"Processing error: {error}")

    with col2:
        if st.button("♻️ Reset Workspace"):
            reset_workspace()
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state["documents_processed"]:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Workspace Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="metric"><div class="metric-label">Documents Uploaded</div>'
            f'<div class="metric-value">{len(st.session_state["document_names"])}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric"><div class="metric-label">Knowledge Chunks</div>'
            f'<div class="metric-value">{st.session_state["total_chunks"]}</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("View uploaded document names"):
        for name in st.session_state["document_names"]:
            st.write(f"- {name}")

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state["documents_processed"] and st.session_state["vector_store"] is not None:
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "💬 Policy Q&A",
            "📄 Document Summary",
            "⚖️ Compare Policies",
            "📋 Rule Extraction",
        ]
    )

    with tab1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💬 Ask About Healthcare Policies</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Answers are grounded in retrieved passages from the uploaded documents.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Clear Chat"):
            st.session_state["chat_history"] = []
            st.rerun()

        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    show_sources(message.get("sources", []))

        question = st.chat_input(
            "Ask about coverage, CPT codes, eligibility, exclusions, or requirements"
        )

        if question:
            st.session_state["chat_history"].append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner(f"Reviewing evidence with {model_choice}..."):
                    try:
                        result = ask_rag(question, model_choice, top_k)
                        st.markdown(result["answer"])
                        show_sources(result.get("source_documents", []))

                        st.session_state["chat_history"].append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result.get("source_documents", []),
                            }
                        )
                    except Exception as error:
                        st.error(f"Answer generation error: {error}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📄 Document Summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Create a concise healthcare-focused summary.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Generate Healthcare Summary"):
            with st.spinner(f"Generating summary with {model_choice}..."):
                try:
                    prompt = """
                    Summarize each uploaded healthcare document. Include:
                    - purpose of the document
                    - covered services or procedures
                    - eligibility or medical-necessity requirements
                    - prior authorization requirements
                    - CPT, HCPCS, or ICD codes
                    - exclusions and limitations
                    - required documentation

                    Do not invent information. State clearly when something is not found.
                    """
                    result = ask_rag(prompt, model_choice, top_k)
                    st.markdown(result["answer"])
                    show_sources(result.get("source_documents", []))
                except Exception as error:
                    st.error(f"Summary error: {error}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚖️ Compare Healthcare Policies</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Compare coverage, authorization, eligibility, exclusions, or documentation.</div>',
            unsafe_allow_html=True,
        )

        request = st.text_input(
            "Comparison request",
            placeholder="Compare prior authorization requirements and coverage limitations.",
        )

        if st.button("Run Policy Comparison"):
            if len(st.session_state["document_names"]) < 2:
                st.warning("Upload at least two documents to compare policies.")
            elif not request.strip():
                st.warning("Enter a comparison request.")
            else:
                with st.spinner(f"Comparing policies with {model_choice}..."):
                    try:
                        result = compare_papers(
                            st.session_state["vector_store"],
                            request,
                            chat_history=st.session_state["chat_history"],
                            per_paper=3,
                            total_k=max(top_k, 15),
                            model_name=model_choice,
                        )
                        st.markdown(result["comparison"])
                        show_sources(result.get("source_documents", []))
                    except Exception as error:
                        st.error(f"Comparison error: {error}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Extract Healthcare Rules</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Extract important policy content using a focused prompt.</div>',
            unsafe_allow_html=True,
        )

        extraction_choice = st.selectbox(
            "Information to extract",
            [
                "CPT / HCPCS / ICD codes",
                "Coverage rules",
                "Prior authorization requirements",
                "Eligibility and medical necessity",
                "Exclusions and limitations",
                "Required documentation",
            ],
        )

        extraction_prompts = {
            "CPT / HCPCS / ICD codes": (
                "Extract every CPT, HCPCS, and ICD code. For each code, provide "
                "its description, related service, document name, and page number."
            ),
            "Coverage rules": (
                "Extract all coverage rules. Include covered services, conditions "
                "for coverage, exceptions, document name, and page number."
            ),
            "Prior authorization requirements": (
                "Extract all prior authorization requirements. Include the service, "
                "requirement, exception, document name, and page number."
            ),
            "Eligibility and medical necessity": (
                "Extract all eligibility criteria and medical-necessity requirements. "
                "Organize them by document and include page references."
            ),
            "Exclusions and limitations": (
                "Extract all exclusions, non-covered services, limitations, exceptions, "
                "and restrictions. Include document and page references."
            ),
            "Required documentation": (
                "Extract all required clinical records, forms, test results, and other "
                "supporting documentation. Include document and page references."
            ),
        }

        if st.button("Extract Information"):
            with st.spinner(f"Extracting information with {model_choice}..."):
                try:
                    result = ask_rag(
                        extraction_prompts[extraction_choice],
                        model_choice,
                        top_k,
                    )
                    st.markdown(result["answer"])
                    show_sources(result.get("source_documents", []))
                except Exception as error:
                    st.error(f"Extraction error: {error}")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload healthcare PDF documents and click **Process Documents** to begin.")

st.markdown(
    '<div class="footer">Healthcare Content Management Assistant • Generative AI + RAG • Evidence-grounded policy analysis</div>',
    unsafe_allow_html=True,
)