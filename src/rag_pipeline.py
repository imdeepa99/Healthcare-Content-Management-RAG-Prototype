from collections import defaultdict
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.prompts import SYSTEM_PROMPT, COMPARISON_PROMPT

load_dotenv()


def get_llm(model_name="ollama"):
    """
    Select which LLM to use:
    - ollama: local model
    - gemini: Google Gemini API
    - groq: Groq LLaMA API
    """

    if model_name == "ollama":
        return ChatOllama(
            model="llama3",
            temperature=0
        )

    elif model_name == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )

    elif model_name == "groq":
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2
        )

    else:
        raise ValueError("Invalid model_name. Choose: ollama, gemini, or groq.")


def format_context(docs):
    context_parts = []

    for doc in docs:
        paper_name = doc.metadata.get("paper_name", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "Unknown")
        text = doc.page_content

        context_parts.append(
            f"[Paper: {paper_name}, Page: {page}, Chunk: {chunk_id}]\n{text}"
        )

    return "\n\n".join(context_parts)


def retrieve_balanced_documents(vector_store, query: str, per_paper: int = 2, total_k: int = 12):
    """
    Retrieve documents and balance them across papers so one paper
    does not dominate the answer.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": total_k})
    docs = retriever.invoke(query)

    grouped = defaultdict(list)

    for doc in docs:
        paper_name = doc.metadata.get("paper_name", "Unknown")
        grouped[paper_name].append(doc)

    balanced_docs = []

    for _, paper_docs in grouped.items():
        balanced_docs.extend(paper_docs[:per_paper])

    return balanced_docs


def build_history_text(chat_history, max_turns: int = 6):
    """
    Convert recent chat history into plain text for prompt context.
    """
    if not chat_history:
        return ""

    recent_history = chat_history[-max_turns:]
    history_parts = []

    for msg in recent_history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        history_parts.append(f"{role}: {content}")

    return "\n".join(history_parts)


def answer_question(
    vector_store,
    question: str,
    chat_history=None,
    k=None,
    per_paper: int = 2,
    total_k: int = 12,
    model_name: str = "ollama"
):
    docs = retrieve_balanced_documents(
        vector_store=vector_store,
        query=question,
        per_paper=per_paper,
        total_k=total_k
    )

    context = format_context(docs)
    history_text = build_history_text(chat_history)

    prompt = f"""
{SYSTEM_PROMPT}

Conversation History:
{history_text}

Important:
- Use the conversation history to resolve follow-up questions like "it", "that paper", "the second one", or "that method".
- Try to mention each uploaded paper if relevant context is available.
- If one uploaded paper is missing from the retrieved context, say that clearly.
- Do not pretend all papers discuss the same thing unless the context supports it.
- Do not invent facts.

Question:
{question}

Retrieved Context:
{context}

Answer:
"""

    llm = get_llm(model_name)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "source_documents": docs
    }


def compare_papers(
    vector_store,
    comparison_request: str,
    chat_history=None,
    k=None,
    per_paper: int = 3,
    total_k: int = 15,
    model_name: str = "ollama"
):
    docs = retrieve_balanced_documents(
        vector_store=vector_store,
        query=comparison_request,
        per_paper=per_paper,
        total_k=total_k
    )

    context = format_context(docs)
    history_text = build_history_text(chat_history)

    prompt = f"""
{COMPARISON_PROMPT}

Conversation History:
{history_text}

Important:
- Compare papers individually.
- Make sure each uploaded paper is considered if context exists.
- If a paper has insufficient retrieved context, mention that explicitly.
- Do not invent missing details.

Comparison Request:
{comparison_request}

Retrieved Context:
{context}

Structured Comparison:
"""

    llm = get_llm(model_name)
    response = llm.invoke(prompt)

    return {
        "comparison": response.content,
        "source_documents": docs
    }


def summarize_each_paper(
    vector_store,
    paper_names,
    model_name: str = "ollama"
):
    llm = get_llm(model_name)
    summaries = []

    for paper_name in paper_names:
        retriever = vector_store.as_retriever(search_kwargs={"k": 8})
        docs = retriever.invoke(f"Summarize the paper {paper_name}")

        filtered_docs = [
            doc for doc in docs
            if doc.metadata.get("paper_name", "") == paper_name
        ]

        if not filtered_docs:
            summaries.append(f"### {paper_name}\nNo relevant context retrieved.")
            continue

        context = format_context(filtered_docs[:4])

        prompt = f"""
{SYSTEM_PROMPT}

Summarize this paper with:
- objective
- method
- dataset
- findings

Paper name:
{paper_name}

Context:
{context}

Summary:
"""

        response = llm.invoke(prompt)
        summaries.append(f"### {paper_name}\n{response.content}")

    return "\n\n".join(summaries)


def generate_comparison_table(
    vector_store,
    paper_names,
    per_paper: int = 3,
    total_k: int = 18,
    model_name: str = "ollama"
):
    """
    Generate a structured JSON comparison table with one row per paper.
    """
    query = (
        "For each uploaded paper, identify its objective, method, dataset, "
        "evaluation metric, main result, and limitation."
    )

    docs = retrieve_balanced_documents(
        vector_store=vector_store,
        query=query,
        per_paper=per_paper,
        total_k=total_k
    )

    context = format_context(docs)
    expected_papers = ", ".join(paper_names)

    prompt = f"""
You are an academic information extraction assistant.

Using only the retrieved context, create a comparison table for the uploaded papers.

Expected paper names:
{expected_papers}

Return ONLY valid JSON.
Do not add explanations.
Do not wrap the JSON in markdown unless absolutely necessary.

Use this exact format:
[
  {{
    "paper_name": "paper1.pdf",
    "objective": "",
    "method": "",
    "dataset": "",
    "metric": "",
    "result": "",
    "limitation": ""
  }}
]

Rules:
- Include one object per paper if evidence exists.
- Use the paper filename exactly as shown when possible.
- If some field is missing, use "Not clearly stated".
- Do not invent details.
- Use only the retrieved context.

Retrieved Context:
{context}

JSON:
"""

    llm = get_llm(model_name)
    response = llm.invoke(prompt)

    return {
        "table_json": response.content,
        "source_documents": docs
    }