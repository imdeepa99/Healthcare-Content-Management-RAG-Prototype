import json
import pandas as pd


def build_source_table(source_documents):
    rows = []

    for doc in source_documents:
        rows.append({
            "Paper Name": doc.metadata.get("paper_name", "Unknown"),
            "Page": doc.metadata.get("page", "Unknown"),
            "Chunk ID": doc.metadata.get("chunk_id", "Unknown"),
            "Preview": doc.page_content[:250].replace("\n", " ")
        })

    return pd.DataFrame(rows)


def parse_comparison_json(raw_text: str) -> pd.DataFrame:
    """
    Parse LLM JSON output into a dataframe.
    Returns a fallback dataframe if parsing fails.
    """
    try:
        cleaned = raw_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "", 1).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "", 1).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        data = json.loads(cleaned)

        if isinstance(data, dict) and "papers" in data:
            data = data["papers"]

        if not isinstance(data, list):
            return pd.DataFrame([{
                "Paper Name": "Parsing Error",
                "Objective": "",
                "Method": "",
                "Dataset": "",
                "Metric": "",
                "Result": "",
                "Limitation": raw_text
            }])

        normalized_rows = []
        for item in data:
            normalized_rows.append({
                "Paper Name": item.get("paper_name", ""),
                "Objective": item.get("objective", ""),
                "Method": item.get("method", ""),
                "Dataset": item.get("dataset", ""),
                "Metric": item.get("metric", ""),
                "Result": item.get("result", ""),
                "Limitation": item.get("limitation", "")
            })

        return pd.DataFrame(normalized_rows)

    except Exception:
        return pd.DataFrame([{
            "Paper Name": "Parsing Error",
            "Objective": "",
            "Method": "",
            "Dataset": "",
            "Metric": "",
            "Result": "",
            "Limitation": raw_text
        }])