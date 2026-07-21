from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    """
    Split extracted page text into smaller overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for page_data in pages:
        paper_name = page_data["paper_name"]
        page = page_data["page"]
        text = page_data["text"]

        split_texts = splitter.split_text(text)

        for i, chunk_text in enumerate(split_texts):
            chunks.append({
                "paper_name": paper_name,
                "page": page,
                "chunk_id": f"{paper_name}_p{page}_c{i}",
                "text": chunk_text
            })

    return chunks