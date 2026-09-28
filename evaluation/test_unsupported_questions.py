from sentence_transformers import SentenceTransformer

from src.retrieval.search import load_vector_store, search
from src.generation.generator import generate_answer


test_questions = [
    "What is the company office address?",
    "What is the employee salary?",
    "What restaurants are near the office?",
    "Who is the CEO of NordicTech?"
]


if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    index, chunks = load_vector_store()

    correct = 0

    print("=" * 60)
    print("UNSUPPORTED QUESTION EVALUATION")
    print("=" * 60)

    for question in test_questions:

        results = search(
            question,
            model,
            index,
            chunks,
            top_k=3,
            distance_threshold=1.1
        )

        print("\n" + "-" * 60)
        print("Question:", question)

        if not results:
            answer = (
                "I don't have enough information in the NordicTech "
                "documents to answer that question."
            )

            print("No relevant context retrieved.")
            print("Expected behavior: Refuse to answer")
            print("Actual behavior:", answer)
            print("Result: PASS")

            correct += 1
            continue

        context = "\n".join(
            result["content"]
            for result in results
        )

        answer = generate_answer(
            question,
            context
        )

        print("\nGenerated answer:")
        print(answer)

        refusal_phrases = [
            "don't have enough information",
            "information is not available",
            "not available in the provided documents",
            "not enough information"
        ]

        answer_lower = answer.lower()

        if any(
            phrase in answer_lower
            for phrase in refusal_phrases
        ):
            correct += 1
            result = "PASS"
        else:
            result = "FAIL"

        print("\nExpected behavior: Refuse to answer")
        print("Result:", result)

    total = len(test_questions)
    accuracy = (correct / total) * 100

    print("\n" + "=" * 60)
    print(
        f"Unsupported question accuracy: "
        f"{correct}/{total} ({accuracy:.1f}%)"
    )
    print("=" * 60)

