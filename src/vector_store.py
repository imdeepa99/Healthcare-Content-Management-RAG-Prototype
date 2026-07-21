from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_documents(chunks: List[dict]) -> List[Document]:
    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "paper_name": chunk["paper_name"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"]
                }
            )
        )

    return documents


def create_vector_store(chunks: List[dict]):
    documents = build_documents(chunks)
    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model
    )

    return vector_store