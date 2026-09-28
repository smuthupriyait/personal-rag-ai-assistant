from sentence_transformers import SentenceTransformer

from src.ingestion.load_documents import load_documents
from src.ingestion.chunk_documents import chunk_documents


def create_embeddings(chunks, model):
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = load_documents()
    chunks = chunk_documents(documents)

    chunks_with_embeddings = create_embeddings(chunks, model)

    print("Embeddings created successfully.")
    print("Number of chunks:", len(chunks_with_embeddings))
    print("Embedding size:", len(chunks_with_embeddings[0]["embedding"]))
