def evaluate_refusal_case(item, response, elapsed_seconds):
    correct_refusal = response.get("status") == "insufficient_evidence"

    return {
        "id": item["id"],
        "difficulty": item.get("difficulty"),
        "question": item["question"],
        "expected_status": "insufficient_evidence",
        "actual_status": response.get("status"),
        "correct_refusal": correct_refusal,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def update_refusal_metrics(results):
    correct = sum(result["correct_refusal"] for result in results)
    total = len(results)

    return {
        "refusal_accuracy": correct / total if total else 0,
        "refusal_passed": correct,
        "refusal_total": total,
        "average_refusal_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / total,
            3,
        ) if total else 0,
    }
