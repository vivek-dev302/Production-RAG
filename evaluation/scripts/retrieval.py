def evaluate_retrieval_case(item, top_chunks, elapsed_seconds):
    expected_chunk_ids = {
        source["chunk_id"]
        for source in item["expected_sources"]
        if source.get("chunk_id") is not None
    }
    retrieved_chunk_ids = {
        chunk["chunk_id"]
        for chunk in top_chunks
        if chunk.get("chunk_id") is not None
    }

    return {
        "id": item["id"],
        "question": item["question"],
        "expected_chunk_ids": sorted(expected_chunk_ids),
        "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
        "passed": bool(expected_chunk_ids & retrieved_chunk_ids),
        "recall": (
            len(expected_chunk_ids & retrieved_chunk_ids) / len(expected_chunk_ids)
            if expected_chunk_ids else 0
        ),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def update_retrieval_metrics(results, top_k):
    passed = sum(result["passed"] for result in results)
    total = len(results)

    return {
        f"hit_rate_at_{top_k}": passed / total if total else 0,
        f"mean_recall_at_{top_k}": round(
            sum(result.get("recall", float(result["passed"])) for result in results) / total,
            3,
        ) if total else 0,
        "hit_count": passed,
        "total": total,
        "average_retrieval_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / total,
            3,
        ) if total else 0,
    }
