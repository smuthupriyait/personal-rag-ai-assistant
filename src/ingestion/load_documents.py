from pathlib import Path


def load_documents():
    documents_path = Path("data/documents")
    documents = []

    for file_path in documents_path.glob("*.txt"):
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