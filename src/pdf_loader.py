import fitz
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str, paper_name: str) -> List[Dict]:
    """
    Extract text page by page from a PDF.
    Returns a list of dictionaries with page text and metadata.
    """
    doc = fitz.open(pdf_path)
    extracted_pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = text.strip()

        if text:
            extracted_pages.append({
                "paper_name": paper_name,
                "page": page_number,
                "text": text
            })

    doc.close()
    return extracted_pages