import faiss
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer

from src.ingestion.load_documents import load_documents
from src.ingestion.chunk_documents import chunk_documents


def create_vector_store(chunks, model):
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = load_documents()
    chunks = chunk_documents(documents)

    index = create_vector_store(chunks, model)

    faiss.write_index(index, "vector_store/index.faiss")

    with open("vector_store/chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)

    print("FAISS vector store created successfully.")
    print("Number of vectors:", index.ntotal)
    print("Vector dimension:", index.d)
    print("Index saved to: vector_store/index.faiss")
    print("Chunks saved to: vector_store/chunks.pkl")