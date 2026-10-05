import json
from pathlib import Path
from time import perf_counter
from app.services.rag import retrieve_chunks


REPORT_PATH = Path("data/evaluation/results/retrieval_report.json")

DATASET_PATH = Path("data/evaluation/golden_dataset.json")


def evaluate_retrieval_case(item, top_chunks, elapsed_seconds):
    expected_chunk_ids = {
        source["chunk_id"]
        for source in item["expected_sources"]
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
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def run_evaluation():
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        report = {
            "run_status": "in_progress",
            "metrics": {},
            "results": [],
        }

    completed_ids = {
        result["id"]
        for result in report["results"]
    }

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    total = 0
    passed = 0

    for item in dataset:
        if item["id"] in completed_ids:
            print(f"[SKIP] {item['id']} already evaluated")
            continue

        if not item["answerable"]:
            continue

        total += 1

        start_time = perf_counter()

        top_chunks = retrieve_chunks(item["question"])

        elapsed_seconds = perf_counter() - start_time

        result = evaluate_retrieval_case(item, top_chunks, elapsed_seconds)
        expected_chunk_ids = result["expected_chunk_ids"]
        retrieved_chunk_ids = result["retrieved_chunk_ids"]
        is_pass = result["passed"]

        if is_pass:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{status}] {item['id']}")
        print(f"Question: {item['question']}")
        print(f"Expected chunk IDs: {sorted(expected_chunk_ids)}")
        print(f"Retrieved chunk IDs: {sorted(retrieved_chunk_ids)}")
        print()

        report["results"].append(result)

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        REPORT_PATH.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8"
        )

    passed = sum(result["passed"] for result in report["results"])
    total = len(report["results"])
    recall_at_3 = passed / total if total else 0
    average_elapsed_seconds = (
        sum(result["elapsed_seconds"] for result in report["results"]) / total
        if total
        else 0
    )

    report["run_status"] = "complete"
    report["metrics"]["recall_at_3"] = recall_at_3
    report["metrics"]["passed"] = passed
    report["metrics"]["total"] = total
    report["metrics"]["average_retrieval_elapsed_seconds"] = round(
        average_elapsed_seconds, 3
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("-" * 50)
    print(f"Recall@3: {passed}/{total} = {recall_at_3:.2%}")
    print(f"Average retrieval latency: {average_elapsed_seconds:.3f} seconds")


if __name__ == "__main__":
    run_evaluation()
