from sentence_transformers import SentenceTransformer

from src.retrieval.search import load_vector_store, search
from src.generation.generator import generate_answer
from src.config import DISTANCE_THRESHOLD

test_questions = [
    {
        "question": "How many vacation days do employees get?",
        "expected_keywords": ["25", "vacation"]
    },
    {
        "question": "How early should I submit a vacation request?",
        "expected_keywords": ["two weeks"]
    },
    {
        "question": "What happens if I forget my login password?",
        "expected_keywords": ["password reset"]
    },
    {
        "question": "Where should I go when my laptop has a problem?",
        "expected_keywords": ["IT support portal"]
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
        "question": "What should I do if I cannot access my account?",
        "expected_keywords": ["password reset"]
    },
    {
        "question": "How far in advance do I need to plan my holiday?",
        "expected_keywords": ["two weeks"]
    },
    {
        "question": "What should I do if I suspect a security problem?",
        "expected_keywords": ["report", "security", "IT support"]
    },
    {
        "question": "Who should I contact when I have an IT issue?",
        "expected_keywords": ["internal support portal"]
    }
]


if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    index, chunks = load_vector_store()

    correct = 0

    print("=" * 60)
    print("RAG GENERATION EVALUATION")
    print("=" * 60)

    for test in test_questions:

        question = test["question"]
        expected_keywords = test["expected_keywords"]

        results = search(
            question,
            model,
            index,
            chunks,
            top_k=3,
            distance_threshold=DISTANCE_THRESHOLD
        )

        print("\n" + "-" * 60)
        print("Question:", question)
        print("Expected key information:", expected_keywords)

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

        print("\nGenerated answer:")
        print(answer)

        answer_lower = answer.lower()

        missing_keywords = []

        for keyword in expected_keywords:
            if keyword.lower() not in answer_lower:
                missing_keywords.append(keyword)

        if not missing_keywords:
            correct += 1
            result = "PASS"
        else:
            result = "FAIL"

        print("\nMissing key information:", missing_keywords)
        print("Result:", result)

    total = len(test_questions)
    accuracy = (correct / total) * 100

    print("\n" + "=" * 60)
    print(
        "Generation key-information accuracy: "
        f"{correct}/{total} ({accuracy:.1f}%)"
    )
    print("=" * 60)



