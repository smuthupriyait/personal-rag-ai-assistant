from sentence_transformers import SentenceTransformer

from src.retrieval.search import load_vector_store, search


test_questions = [
    {
        "question": "How many vacation days do employees get?",
        "expected_file": "leave_policy.txt",
        "expected_information": "25 paid vacation days"
    },
    {
        "question": "How early should I submit a vacation request?",
        "expected_file": "leave_policy.txt",
        "expected_information": "at least two weeks"
    },
    {
        "question": "What happens if I forget my login password?",
        "expected_file": "it_support_policy.txt",
        "expected_information": "password reset process"
    },
    {
        "question": "Where should I go when my laptop has a problem?",
        "expected_file": "it_support_policy.txt",
        "expected_information": "IT support portal"
    },
    {
        "question": "What are the normal hours employees are expected to work?",
        "expected_file": "employee_handbook.txt",
        "expected_information": "Monday to Friday, 09:00 to 17:00"
    },
    {
        "question": "Am I allowed to work from home?",
        "expected_file": "employee_handbook.txt",
        "expected_information": "work remotely"
    },
    {
        "question": "What should I do if I cannot access my account?",
        "expected_file": "it_support_policy.txt",
        "expected_information": "password reset process"
    },
    {
        "question": "How far in advance do I need to plan my holiday?",
        "expected_file": "leave_policy.txt",
        "expected_information": "at least two weeks"
    },
    {
        "question": "What should I do if I suspect a security problem?",
        "expected_file": "employee_handbook.txt",
        "expected_information": "report suspected security incidents"
    },
    {
        "question": "Who should I contact when I have an IT issue?",
        "expected_file": "it_support_policy.txt",
        "expected_information": "internal support portal"
    }
]


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index, chunks = load_vector_store()

    top1_correct = 0
    top3_correct = 0
    information_correct = 0

    print("=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    for test in test_questions:
        question = test["question"]
        expected_file = test["expected_file"]
        expected_information = test["expected_information"]

        results = search(
            question,
            model,
            index,
            chunks,
            top_k=3
        )

        print("\n" + "-" * 60)
        print("Question:", question)
        print("Expected file:", expected_file)
        print("Expected information:", expected_information)

        if results:
            print("\nRetrieved chunks:")

            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank}. {result['filename']} "
                    f"— {result['section']} "
                    f"(distance: {result['distance']:.4f})"
                )
                print(f"   Content: {result['content']}")

        retrieved_files = [
            result["filename"]
            for result in results
        ]

        # Top-1 evaluation
        if retrieved_files and retrieved_files[0] == expected_file:
            top1_correct += 1
            top1_result = "PASS"
        else:
            top1_result = "FAIL"

        # Top-3 evaluation
        if expected_file in retrieved_files:
            top3_correct += 1
            top3_result = "PASS"
        else:
            top3_result = "FAIL"

        # Information evaluation
        information_found = any(
            expected_information.lower() in result["content"].lower()
            for result in results
        )

        if information_found:
            information_correct += 1
            information_result = "PASS"
        else:
            information_result = "FAIL"

        print("\nTop-1:", top1_result)
        print("Top-3:", top3_result)
        print("Information:", information_result)

    total = len(test_questions)

    top1_accuracy = (top1_correct / total) * 100
    top3_accuracy = (top3_correct / total) * 100
    information_accuracy = (information_correct / total) * 100

    print("\n" + "=" * 60)
    print(
        f"Top-1 accuracy: {top1_correct}/{total} "
        f"({top1_accuracy:.1f}%)"
    )
    print(
        f"Top-3 accuracy: {top3_correct}/{total} "
        f"({top3_accuracy:.1f}%)"
    )
    print(
        f"Information accuracy: {information_correct}/{total} "
        f"({information_accuracy:.1f}%)"
    )
    print("=" * 60)