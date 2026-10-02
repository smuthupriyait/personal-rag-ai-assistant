from sentence_transformers import SentenceTransformer

from src.retrieval.search import load_vector_store, search
from src.generation.generator import generate_answer
from src.config import DISTANCE_THRESHOLD, EMBEDDING_MODEL


test_questions = [
    {
        "question": "How many vacation days do employees get?",
        "expected_keywords": ["25", "vacation"]
    },
    {
        "question": "What are the normal hours employees are expected to work?",
        "expected_keywords": ["Monday to Friday", "09:00", "17:00"]
    },
    {
        "question": "Am I allowed to work from home?",
        "expected_keywords": ["work remotely"]
    },
    {
        "question": "What happens if I forget my login password?",
        "expected_keywords": ["password reset"]
    },
    {
        "question": "What is the company office address?",
        "expected_refusal": True
    },
    {
        "question": "Who is the CEO of NordicTech?",
        "expected_refusal": True
    }
]


if __name__ == "__main__":

    model = SentenceTransformer(EMBEDDING_MODEL)

    index, chunks = load_vector_store()

    correct = 0

    print("=" * 60)
    print("END-TO-END RAG REGRESSION TEST")
    print("=" * 60)

    for test in test_questions:

        question = test["question"]

        print("\n" + "-" * 60)
        print("Question:", question)

        results = search(
            question,
            model,
            index,
            chunks,
            top_k=3,
            distance_threshold=DISTANCE_THRESHOLD
        )

        if test.get("expected_refusal"):

            if not results:
                print("No relevant context retrieved.")
                print("Expected: Refuse to answer")
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

            print("Generated answer:")
            print(answer)

            refusal_phrases = [
                "don't have enough information",
                "information is not available",
                "not available in the provided documents",
                "not enough information",
                "does not contain any information"
            ]

            if any(
                phrase in answer.lower()
                for phrase in refusal_phrases
            ):
                print("Expected: Refuse to answer")
                print("Result: PASS")
                correct += 1
            else:
                print("Expected: Refuse to answer")
                print("Result: FAIL")

            continue

        if not results:
            print("No relevant context retrieved.")
            print("Result: FAIL")
            continue

        context = "\n".join(
            result["content"]
            for result in results
        )

        answer = generate_answer(
            question,
            context
        )

        print("Generated answer:")
        print(answer)

        answer_lower = answer.lower()

        missing_keywords = []

        for keyword in test["expected_keywords"]:
            if keyword.lower() not in answer_lower:
                missing_keywords.append(keyword)

        if not missing_keywords:
            print("Result: PASS")
            correct += 1
        else:
            print("Missing key information:", missing_keywords)
            print("Result: FAIL")

    total = len(test_questions)
    accuracy = (correct / total) * 100

    print("\n" + "=" * 60)
    print(
        f"End-to-end accuracy: "
        f"{correct}/{total} ({accuracy:.1f}%)"
    )
    print("=" * 60)
