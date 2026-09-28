from .load_documents import load_documents


def chunk_documents(documents):
    chunks = []

    for document in documents:
        lines = document["content"].splitlines()

        current_section = "General"
        current_content = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith("## "):
                if current_content:
                    chunks.append({
                        "filename": document["filename"],
                        "section": current_section,
                        "content": " ".join(current_content)
                    })

                current_section = line.replace("## ", "")
                current_content = []

            elif not line.startswith("# "):
                current_content.append(line)

        if current_content:
            chunks.append({
                "filename": document["filename"],
                "section": current_section,
                "content": " ".join(current_content)
            })

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    print("Number of chunks:", len(chunks))

    for chunk in chunks:
        print("=" * 50)
        print(f"File: {chunk['filename']}")
        print(f"Section: {chunk['section']}")
        print(chunk["content"])