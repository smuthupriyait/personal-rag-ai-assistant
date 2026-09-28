from sentence_transformers import SentenceTransformer

from src.retrieval.search import load_vector_store, search


test_questions = [
    {
        "question": "How many vacation days do employees get?",
        "expected_file": "leave_policy.txt"
    },
    {
        "question": "How early should I submit a vacation request?",
        "expected_file": "leave_policy.txt"
    },
    {
        "question": "What happens if I forget my login password?",
        "expected_file": "it_support_policy.txt"
    },
    {
        "question": "Where should I go when my laptop has a problem?",
        "expected_file": "it_support_policy.txt"
    },
    {
        "question": "What are the normal hours employees are expected to work?",
        "expected_file": "employee_handbook.txt"
    },
    {
        "question": "Am I allowed to work from home?",
        "expected_file": "employee_handbook.txt"
    },
    {
        "question": "What should I do if I cannot access my account?",
        "expected_file": "it_support_policy.txt"
    },
    {
        "question": "How far in advance do I need to plan my holiday?",
        "expected_file": "leave_policy.txt"
    },
    {
        "question": "What should I do if I suspect a security problem?",
        "expected_file": "employee_handbook.txt"
    },
    {
        "question": "Who should I contact when I have an IT issue?",
        "expected_file": "it_support_policy.txt"
    }
]


thresholds = [1.0, 1.1, 1.2]


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index, chunks = load_vector_store()

    print("=" * 70)
    print("RETRIEVAL THRESHOLD COMPARISON")
    print("=" * 70)

    for threshold in thresholds:
        correct = 0

        print("\n" + "=" * 70)
        print("THRESHOLD:", threshold)
        print("=" * 70)

        for test in test_questions:
            results = search(
                test["question"],
                model,
                index,
                chunks,
                top_k=3,
                distance_threshold=threshold
            )

            retrieved_files = [
                result["filename"]
                for result in results
            ]

            if test["expected_file"] in retrieved_files:
                correct += 1
                result = "PASS"
            else:
                result = "FAIL"

            print(
                f"{result} | "
                f"{test['question']} | "
                f"Retrieved: {retrieved_files}"
            )

        accuracy = (correct / len(test_questions)) * 100

        print(
            f"\nThreshold {threshold}: "
            f"{correct}/{len(test_questions)} "
            f"({accuracy:.1f}%)"
        )

