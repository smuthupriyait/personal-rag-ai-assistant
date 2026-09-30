from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"


def load_vector_store():
    index = faiss.read_index(
        str(VECTOR_STORE_PATH / "index.faiss")
    )

    with open(VECTOR_STORE_PATH / "chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


def search(query, model, index, chunks, top_k=3, distance_threshold=None):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, index_position in zip(distances[0], indices[0]):
        if index_position == -1:
            continue

        if distance_threshold is not None and distance > distance_threshold:
            continue

        results.append({
            "filename": chunks[index_position]["filename"],
            "section": chunks[index_position]["section"],
            "content": chunks[index_position]["content"],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index, chunks = load_vector_store()

    query = "Am I allowed to work from home?"

    results = search(
        query,
        model,
        index,
        chunks,
        top_k=5
    )

    print("=" * 60)
    print("RETRIEVAL THRESHOLD DIAGNOSTIC")
    print("=" * 60)

    print("\nQuery:", query)

    for result in results:
        print("\n" + "-" * 60)
        print("File:", result["filename"])
        print("Section:", result["section"])
        print("Distance:", result["distance"])
        print("Content:", result["content"])

