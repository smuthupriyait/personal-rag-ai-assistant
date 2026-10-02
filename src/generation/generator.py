import logging

from sentence_transformers import SentenceTransformer
from ollama import chat

from src.config import DISTANCE_THRESHOLD, OLLAMA_MODEL, EMBEDDING_MODEL
from src.logging_config import setup_logging
from src.retrieval.search import load_vector_store, search


def prepare_context(retrieved_chunks):
    return "\n".join(
        chunk["content"] for chunk in retrieved_chunks
    )


def generate_answer(question, context):
    try:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"""
Answer the question using only the provided context.

Give a clear, complete sentence as the answer.
Do not answer with only a number or a few words.
Do not add information that is not present in the context.

If the context does not contain enough information to answer the question,
say that the information is not available in the provided documents.

Context:
{context}

Question:
{question}
"""
                }
            ]
        )

        return response["message"]["content"]

    except Exception as error:
        logging.error("Failed to generate answer: %s", error)
        return (
            "I'm unable to generate an answer right now. "
            "Please try again later."
        )


if __name__ == "__main__":
    setup_logging()

    model = SentenceTransformer(EMBEDDING_MODEL)

    index, chunks = load_vector_store()

    print("=" * 50)
    print("NordicTech RAG Assistant")
    print("Type 'exit' to stop.")
    print("=" * 50)

    while True:
        question = input("\nAsk a question: ")

        logging.info("Question received: %s", question)

        if question.lower() == "exit":
            print("Goodbye!")
            break

        retrieved_chunks = search(
            question,
            model,
            index,
            chunks,
            distance_threshold=DISTANCE_THRESHOLD
        )

        logging.info("Retrieved %d chunks", len(retrieved_chunks))

        if retrieved_chunks:
            logging.info(
                "Best retrieval distance: %.4f",
                retrieved_chunks[0]["distance"]
            )

            logging.info(
                "Best source: %s — %s",
                retrieved_chunks[0]["filename"],
                retrieved_chunks[0]["section"]
            )

        if not retrieved_chunks:
            print("\nAnswer:")
            print(
                "I don't have enough information in the NordicTech "
                "documents to answer that question."
            )
            continue

        context = prepare_context(retrieved_chunks)

        answer = generate_answer(
            question,
            context
        )

        logging.info("Answer generated successfully")

        print("\nAnswer:")
        print(answer)

        print("\nSource:")
        for chunk in retrieved_chunks:
            print(
                f"- {chunk['filename']} — {chunk['section']}"
            )