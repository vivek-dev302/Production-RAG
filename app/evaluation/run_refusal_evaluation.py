import json
from pathlib import Path
from time import perf_counter

from app.services.rag import ask_documents


DATASET_PATH = Path("data/evaluation/golden_dataset.json")
REPORT_PATH = Path("data/evaluation/results/refusal_report.json")

def save_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_evaluation() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        report = {"metrics": {}, "refusal_results+": []}

    report["run_status"] = "in_progress"

    completed_ids = {
        result["id"]
        for result in report["refusal_results"]
    }

    for item in dataset:
        if item["answerable"]:
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

        correct_refusal = response.get("status") == "insufficient_evidence"
        status = "PASS" if correct_refusal else "FAIL"

        result = {
            "id": item["id"],
            "difficulty": item.get("difficulty"),
            "question": item["question"],
            "expected_status": "insufficient_evidence",
            "actual_status": response.get("status"),
            "correct_refusal": correct_refusal,
            "elapsed_seconds": round(elapsed_seconds, 3),
        }

        report["refusal_results"].append(result)
        save_report(report)

        print(f"[{status}] {item['id']}")
        print(f"Question: {item['question']}")
        print(f"Expected: insufficient_evidence")
        print(f"Actual: {response.get('status')}")
        print()

    refusal_results = report["refusal_results"]
    correct = sum(result["correct_refusal"] for result in refusal_results)
    total = len(refusal_results)
    refusal_accuracy = correct / total if total else 0
    average_elapsed_seconds = (
        sum(result["elapsed_seconds"] for result in refusal_results) / total
        if total
        else 0
    )

    report["metrics"]["refusal_accuracy"] = refusal_accuracy
    report["metrics"]["refusal_passed"] = correct
    report["metrics"]["refusal_total"] = total
    report["metrics"]["average_refusal_elapsed_seconds"] = round(
        average_elapsed_seconds, 3
    )

    if total == sum(not item["answerable"] for item in dataset):
        report["run_status"] = "complete"

    save_report(report)

    print("-" * 50)
    print(f"Refusal accuracy: {correct}/{total} = {refusal_accuracy:.2%}")
    print(f"Average refusal latency: {average_elapsed_seconds:.3f} seconds")


if __name__ == "__main__":
    run_evaluation()
