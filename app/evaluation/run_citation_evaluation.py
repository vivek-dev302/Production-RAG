import json
from pathlib import Path
from time import perf_counter

from app.services.rag import ask_documents


DATASET_PATH = Path("data/evaluation/golden_dataset.json")
REPORT_PATH = Path("data/evaluation/results/citation_report.json")


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


def save_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_evaluation() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        report = {"metrics": {}, "citation_results": []}

    report["run_status"] = "in_progress"

    completed_ids = {
        result["id"]
        for result in report["citation_results"]
    }

    for item in dataset:
        if not item["answerable"]:
            continue

        if item["id"] in completed_ids:
            print(f"[SKIP] {item['id']} already evaluated")
            continue

        try:
            start_time = perf_counter()
            response = ask_documents(item["question"])
            elapsed_seconds = perf_counter() - start_time
        except Exception as error:
            report["run_status"] = "paused_due_to_error"
            report["last_error"] = str(error)
            save_report(report)
            print(f"[STOPPED] {item['id']}: {error}")
            break

        result = evaluate_citation_case(item, response, elapsed_seconds)
        citation_valid = result["citation_valid"]
        status = "PASS" if citation_valid else "FAIL"

        report["citation_results"].append(result)
        save_report(report)

        print(f"[{status}] {item['id']}")
        print(f"Question: {item['question']}")
        print(f"Status: {response.get('status')} | Citations: {result['citation_count']}")
        print()

    citation_results = report["citation_results"]
    valid = sum(result["citation_valid"] for result in citation_results)
    total = len(citation_results)
    citation_validity = valid / total if total else 0
    average_elapsed_seconds = (
        sum(result["elapsed_seconds"] for result in citation_results) / total
        if total
        else 0
    )

    report["metrics"]["citation_validity"] = citation_validity
    report["metrics"]["citation_valid_passed"] = valid
    report["metrics"]["citation_valid_total"] = total
    report["metrics"]["average_citation_elapsed_seconds"] = round(
        average_elapsed_seconds, 3
    )

    if total == sum(item["answerable"] for item in dataset):
        report["run_status"] = "complete"

    save_report(report)

    print("-" * 50)
    print(f"Citation validity: {valid}/{total} = {citation_validity:.2%}")
    print(f"Average citation latency: {average_elapsed_seconds:.3f} seconds")


if __name__ == "__main__":
    run_evaluation()
