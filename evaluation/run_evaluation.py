import json
from pathlib import Path
from time import perf_counter

from app.services.rag import answer_from_chunks, retrieve_chunk_sets
from evaluation.scripts.citation import evaluate_citation_case, update_citation_metrics
from evaluation.scripts.ragas_metrics import (
    create_ragas_metrics,
    evaluate_ragas_case,
    update_ragas_metrics,
)
from evaluation.scripts.refusal import evaluate_refusal_case, update_refusal_metrics
from evaluation.scripts.retrieval import evaluate_retrieval_case, update_retrieval_metrics


DATASET_PATH = Path("data/evaluation/golden_dataset.json")
RESULTS_DIRECTORY = Path("data/evaluation/results")


def load_report(name, result_key):
    path = RESULTS_DIRECTORY / name
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        report = {"metrics": {}, result_key: []}

    report.setdefault("metrics", {})
    report.setdefault(result_key, [])
    return path, report


def save_reports(reports):
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path, report in reports:
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def upsert_result(results, result):
    for index, existing_result in enumerate(results):
        if existing_result["id"] == result["id"]:
            results[index] = result
            return
    results.append(result)


def completed_ids(*result_lists):
    return set.intersection(*[{result["id"] for result in results} for results in result_lists])


def run_evaluation():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    answerable_cases = [item for item in dataset if item["answerable"]]
    unanswerable_cases = [item for item in dataset if not item["answerable"]]

    retrieval_path, retrieval_report = load_report("retrieval_report.json", "results")
    bm25_path, bm25_report = load_report("bm25_retrieval_report.json", "results")
    semantic_path, semantic_report = load_report(
        "semantic_retrieval_report.json", "results"
    )
    citation_path, citation_report = load_report("citation_report.json", "citation_results")
    refusal_path, refusal_report = load_report("refusal_report.json", "refusal_results")
    faithfulness_path, faithfulness_report = load_report(
        "faithfulness_report.json", "faithfulness_results"
    )
    correctness_path, correctness_report = load_report(
        "answer_correctness_report.json", "answer_correctness_results"
    )
    reports = [
        (retrieval_path, retrieval_report),
        (bm25_path, bm25_report),
        (semantic_path, semantic_report),
        (citation_path, citation_report),
        (refusal_path, refusal_report),
        (faithfulness_path, faithfulness_report),
        (correctness_path, correctness_report),
    ]

    for _, report in reports:
        report["run_status"] = "in_progress"

    completed_answerable = completed_ids(
        retrieval_report["results"],
        bm25_report["results"],
        semantic_report["results"],
        citation_report["citation_results"],
        faithfulness_report["faithfulness_results"],
        correctness_report["answer_correctness_results"],
    )
    completed_refusal = {result["id"] for result in refusal_report["refusal_results"]}

    faithfulness_metric, correctness_metric = create_ragas_metrics()
    stopped = False

    for item in answerable_cases:
        if item["id"] in completed_answerable:
            print(f"[SKIP] {item['id']} already evaluated")
            continue

        try:
            retrieval_start = perf_counter()
            chunk_sets = retrieve_chunk_sets(item["question"])
            retrieval_elapsed_seconds = perf_counter() - retrieval_start
            bm25_chunks = chunk_sets["bm25_chunks"]
            semantic_chunks = chunk_sets["semantic_chunks"]
            top_chunks = chunk_sets["reranked_chunks"]

            answer_start = perf_counter()
            response = answer_from_chunks(item["question"], top_chunks)
            answer_elapsed_seconds = retrieval_elapsed_seconds + (perf_counter() - answer_start)

            retrieval_result = evaluate_retrieval_case(
                item, top_chunks, retrieval_elapsed_seconds
            )
            bm25_result = evaluate_retrieval_case(
                item, bm25_chunks, retrieval_elapsed_seconds
            )
            semantic_result = evaluate_retrieval_case(
                item, semantic_chunks, retrieval_elapsed_seconds
            )
            citation_result = evaluate_citation_case(
                item, response, answer_elapsed_seconds
            )
            faithfulness_result, correctness_result = evaluate_ragas_case(
                item,
                response,
                top_chunks,
                faithfulness_metric,
                correctness_metric,
            )
        except Exception as error:
            stopped = True
            for _, report in reports:
                report["run_status"] = "paused_due_to_error"
                report["last_error"] = str(error)
            save_reports(reports)
            print(f"[STOPPED] {item['id']}: {error}")
            break

        upsert_result(retrieval_report["results"], retrieval_result)
        upsert_result(bm25_report["results"], bm25_result)
        upsert_result(semantic_report["results"], semantic_result)
        upsert_result(citation_report["citation_results"], citation_result)
        upsert_result(faithfulness_report["faithfulness_results"], faithfulness_result)
        upsert_result(correctness_report["answer_correctness_results"], correctness_result)
        save_reports(reports)

        print(
            f"[{item['id']}] BM25: {'PASS' if bm25_result['passed'] else 'FAIL'} | "
            f"semantic: {'PASS' if semantic_result['passed'] else 'FAIL'} | "
            f"reranked: {'PASS' if retrieval_result['passed'] else 'FAIL'} | "
            f"citation: {'PASS' if citation_result['citation_valid'] else 'FAIL'} | "
            f"faithfulness: {faithfulness_result['faithfulness_score']} | "
            f"correctness: {correctness_result['answer_correctness_score']}"
        )

    if not stopped:
        for item in unanswerable_cases:
            if item["id"] in completed_refusal:
                print(f"[SKIP] {item['id']} already evaluated")
                continue

            try:
                start_time = perf_counter()
                top_chunks = retrieve_chunk_sets(item["question"])["reranked_chunks"]
                response = answer_from_chunks(item["question"], top_chunks)
                elapsed_seconds = perf_counter() - start_time
                refusal_result = evaluate_refusal_case(item, response, elapsed_seconds)
            except Exception as error:
                stopped = True
                for _, report in reports:
                    report["run_status"] = "paused_due_to_error"
                    report["last_error"] = str(error)
                save_reports(reports)
                print(f"[STOPPED] {item['id']}: {error}")
                break

            upsert_result(refusal_report["refusal_results"], refusal_result)
            save_reports(reports)
            print(
                f"[{item['id']}] refusal: "
                f"{'PASS' if refusal_result['correct_refusal'] else 'FAIL'}"
            )

    retrieval_report["metrics"] = update_retrieval_metrics(
        retrieval_report["results"], top_k=3
    )
    bm25_report["metrics"] = update_retrieval_metrics(
        bm25_report["results"], top_k=6
    )
    semantic_report["metrics"] = update_retrieval_metrics(
        semantic_report["results"], top_k=6
    )
    citation_report["metrics"] = update_citation_metrics(citation_report["citation_results"])
    refusal_report["metrics"] = update_refusal_metrics(refusal_report["refusal_results"])
    faithfulness_report["metrics"] = update_ragas_metrics(
        faithfulness_report["faithfulness_results"],
        "faithfulness_score",
        "average_faithfulness",
    )
    correctness_report["metrics"] = update_ragas_metrics(
        correctness_report["answer_correctness_results"],
        "answer_correctness_score",
        "average_answer_correctness",
    )

    if not stopped:
        expected_answerable = len(answerable_cases)
        expected_refusal = len(unanswerable_cases)
        retrieval_report["run_status"] = (
            "complete" if len(retrieval_report["results"]) == expected_answerable else "in_progress"
        )
        bm25_report["run_status"] = (
            "complete" if len(bm25_report["results"]) == expected_answerable else "in_progress"
        )
        semantic_report["run_status"] = (
            "complete" if len(semantic_report["results"]) == expected_answerable else "in_progress"
        )
        citation_report["run_status"] = (
            "complete" if len(citation_report["citation_results"]) == expected_answerable else "in_progress"
        )
        faithfulness_report["run_status"] = (
            "complete" if len(faithfulness_report["faithfulness_results"]) == expected_answerable else "in_progress"
        )
        correctness_report["run_status"] = (
            "complete" if len(correctness_report["answer_correctness_results"]) == expected_answerable else "in_progress"
        )
        refusal_report["run_status"] = (
            "complete" if len(refusal_report["refusal_results"]) == expected_refusal else "in_progress"
        )

    save_reports(reports)


if __name__ == "__main__":
    run_evaluation()
