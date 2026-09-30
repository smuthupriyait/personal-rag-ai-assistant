from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.load_documents import load_documents
from src.ingestion.chunk_documents import chunk_documents


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"


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

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
       str(VECTOR_STORE_PATH / "index.faiss") 
    )

    with open(VECTOR_STORE_PATH / "chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)

    print("FAISS vector store created successfully.")
    print("Number of vectors:", index.ntotal)
    print("Vector dimension:", index.d)
    print("Index saved to:", VECTOR_STORE_PATH / "index.faiss")
    print("Chunks saved to:", VECTOR_STORE_PATH / "chunks.pkl")