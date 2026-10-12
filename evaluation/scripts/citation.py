def evaluate_citation_case(item, response, elapsed_seconds):
    citations = response.get("citations", [])
    citation_valid = (
        response.get("status") == "answered"
        and response.get("is_cited") is True
        and len(citations) > 0
    )

    return {
        "id": item["id"],
        "question": item["question"],
        "expected_status": "answered",
        "actual_status": response.get("status"),
        "is_cited": response.get("is_cited", False),
        "citation_count": len(citations),
        "citation_valid": citation_valid,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def update_citation_metrics(results):
    valid = sum(result["citation_valid"] for result in results)
    total = len(results)

    return {
        "citation_validity": valid / total if total else 0,
        "citation_valid_passed": valid,
        "citation_valid_total": total,
        "average_citation_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / total,
            3,
        ) if total else 0,
    }
