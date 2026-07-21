# Healthcare Content Management Assistant using RAG

## Overview

This project is a proof-of-concept Healthcare Content Management Assistant developed using Generative AI and Retrieval-Augmented Generation (RAG).

The application allows users to upload healthcare policy documents and interact with them through natural language. Instead of manually reviewing lengthy documents, users can quickly retrieve information, generate summaries, compare policies, and extract important healthcare rules.

---

## Features

- Upload healthcare PDF documents
- Policy Question & Answer
- Document Summary
- Compare Healthcare Policies
- Rule Extraction
- Evidence-based responses using RAG

---

## Technology Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- ChromaDB
- Sentence Transformers
- PyMuPDF

---

## How It Works

1. Upload one or more healthcare PDF documents.
2. Process the documents to create vector embeddings.
3. Store embeddings in ChromaDB.
4. Ask questions or select a feature.
5. The system retrieves relevant document sections.
6. Gemini generates an evidence-based response using the retrieved content.

---

## Project Structure

```
app.py                 # Main Streamlit application
src/                   # Application modules
requirements.txt       # Project dependencies
README.md              # Project documentation
```

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd Healthcare-Content-Management-RAG-Prototype
```

Install the required packages.

```bash
pip install -r requirements.txt
```

Run the application.

```bash
streamlit run app.py
```

---

## Future Improvements

- Support additional healthcare document formats
- Multi-document search
- User authentication
- Cloud deployment
- Improved policy comparison

---

## Author

**Dipa Khadka**

Healthcare Content Management Assistant using Generative AI and Retrieval-Augmented Generation (RAG).