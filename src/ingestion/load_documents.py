from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents"


def load_documents():
    documents = []

    for file_path in DOCUMENTS_PATH.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")

        documents.append({
            "filename": file_path.name,
            "content": content
        })

    return documents


if __name__ == "__main__":
    documents = load_documents()

    for document in documents:
        print("=" * 50)
        print(f"File: {document['filename']}")
        print(document["content"])