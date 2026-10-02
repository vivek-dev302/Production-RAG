import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import Faithfulness

from app.evaluation.run_citation_evaluation import evaluate_citation_case
from app.evaluation.run_refusal_evaluation import evaluate_refusal_case
from app.evaluation.run_retrieval_evaluation import evaluate_retrieval_case
from app.services.rag import answer_from_chunks, retrieve_chunks


load_dotenv()

DATASET_PATH = Path("data/evaluation/golden_dataset.json")
RETRIEVAL_REPORT_PATH = Path("data/evaluation/results/retrieval_report.json")
CITATION_REPORT_PATH = Path("data/evaluation/results/citation_report.json")
REFUSAL_REPORT_PATH = Path("data/evaluation/results/refusal_report.json")
FAITHFULNESS_REPORT_PATH = Path("data/evaluation/results/faithfulness_report.json")
CORRECTNESS_REPORT_PATH = Path(
    "data/evaluation/results/answer_correctness_report.json"
)


def load_report(path: Path, result_key: str) -> dict:
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        report = {"metrics": {}}

    report.setdefault("metrics", {})
    report.setdefault(result_key, [])
    return report


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def upsert_result(results: list[dict], result: dict) -> None:
    for index, existing_result in enumerate(results):
        if existing_result["id"] == result["id"]:
            results[index] = result
            return

    results.append(result)


def create_ragas_metrics() -> tuple[Faithfulness, AnswerCorrectness]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the .env file.")

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=os.getenv("RAGAS_EVALUATOR_MODEL", "gemini-3.5-flash"),
            google_api_key=api_key,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
    )

    return (
        Faithfulness(llm=evaluator_llm),
        AnswerCorrectness(
            llm=evaluator_llm,
            embeddings=embeddings,
            answer_similarity=AnswerSimilarity(embeddings=embeddings),
        ),
    )


def update_retrieval_metrics(report: dict) -> None:
    results = report["results"]
    passed = sum(result["passed"] for result in results)
    total = len(results)

    report["metrics"] = {
        "recall_at_3": passed / total if total else 0,
        "passed": passed,
        "total": total,
        "average_retrieval_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / total,
            3,
        ) if total else 0,
    }


def update_citation_metrics(report: dict) -> None:
    results = report["citation_results"]
    valid = sum(result["citation_valid"] for result in results)
    total = len(results)

    report["metrics"] = {
        "citation_validity": valid / total if total else 0,
        "citation_valid_passed": valid,
        "citation_valid_total": total,
        "average_citation_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / total,
            3,
        ) if total else 0,
    }


def update_refusal_metrics(report: dict) -> None:
    results = report["refusal_results"]
    correct = sum(result["correct_refusal"] for result in results)
    total = len(results)

    report["metrics"] = {
        "refusal_accuracy": correct / total if total else 0,
        "refusal_passed": correct,
        "refusal_total": total,
        "average_refusal_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / total,
            3,
        ) if total else 0,
    }


def update_ragas_metrics(report: dict, result_key: str, score_key: str, metric_name: str) -> None:
    results = report[result_key]
    scored_results = [result for result in results if result[score_key] is not None]

    report["metrics"] = {
        metric_name: round(
            sum(result[score_key] for result in scored_results) / len(scored_results),
            3,
        ) if scored_results else 0,
        "scored_cases": len(scored_results),
        "evaluated_cases": len(results),
        "average_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / len(results),
            3,
        ) if results else 0,
    }


def save_all_reports(reports: list[tuple[Path, dict]]) -> None:
    for path, report in reports:
        save_report(path, report)


def run_evaluation() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    answerable_cases = [item for item in dataset if item["answerable"]]
    unanswerable_cases = [item for item in dataset if not item["answerable"]]

    retrieval_report = load_report(RETRIEVAL_REPORT_PATH, "results")
    citation_report = load_report(CITATION_REPORT_PATH, "citation_results")
    refusal_report = load_report(REFUSAL_REPORT_PATH, "refusal_results")
    faithfulness_report = load_report(
        FAITHFULNESS_REPORT_PATH,
        "faithfulness_results",
    )
    correctness_report = load_report(
        CORRECTNESS_REPORT_PATH,
        "answer_correctness_results",
    )

    reports = [
        (RETRIEVAL_REPORT_PATH, retrieval_report),
        (CITATION_REPORT_PATH, citation_report),
        (REFUSAL_REPORT_PATH, refusal_report),
        (FAITHFULNESS_REPORT_PATH, faithfulness_report),
        (CORRECTNESS_REPORT_PATH, correctness_report),
    ]
    for _, report in reports:
        report["run_status"] = "in_progress"

    completed_answerable_ids = set.intersection(
        {result["id"] for result in retrieval_report["results"]},
        {result["id"] for result in citation_report["citation_results"]},
        {result["id"] for result in faithfulness_report["faithfulness_results"]},
        {
            result["id"]
            for result in correctness_report["answer_correctness_results"]
        },
    )
    completed_refusal_ids = {
        result["id"] for result in refusal_report["refusal_results"]
    }

    faithfulness_metric, correctness_metric = create_ragas_metrics()
    stopped = False

    for item in answerable_cases:
        if item["id"] in completed_answerable_ids:
            print(f"[SKIP] {item['id']} already evaluated")
            continue

        try:
            retrieval_start = perf_counter()
            top_chunks = retrieve_chunks(item["question"])
            retrieval_elapsed_seconds = perf_counter() - retrieval_start

            rag_start = perf_counter()
            response = answer_from_chunks(item["question"], top_chunks)
            rag_elapsed_seconds = retrieval_elapsed_seconds + (
                perf_counter() - rag_start
            )

            faithfulness_score = None
            answer_correctness_score = None
            if response.get("status") == "answered":
                faithfulness_score = float(
                    faithfulness_metric.single_turn_score(
                        SingleTurnSample(
                            user_input=item["question"],
                            response=response["response"],
                            retrieved_contexts=[
                                chunk["page_content"] for chunk in top_chunks
                            ],
                        )
                    )
                )

                answer_correctness_score = float(
                    correctness_metric.single_turn_score(
                        SingleTurnSample(
                            user_input=item["question"],
                            response=response["response"],
                            reference=item["reference_answer"],
                        )
                    )
                )

            total_elapsed_seconds = perf_counter() - retrieval_start
        except Exception as error:
            stopped = True
            for _, report in reports:
                report["run_status"] = "paused_due_to_error"
                report["last_error"] = str(error)
            save_all_reports(reports)
            print(f"[STOPPED] {item['id']}: {error}")
            break

        upsert_result(
            retrieval_report["results"],
            evaluate_retrieval_case(
                item,
                top_chunks,
                retrieval_elapsed_seconds,
            ),
        )
        upsert_result(
            citation_report["citation_results"],
            evaluate_citation_case(item, response, rag_elapsed_seconds),
        )

        common_ragas_result = {
            "id": item["id"],
            "question": item["question"],
            "actual_status": response.get("status"),
            "retrieved_chunk_ids": [chunk.get("chunk_id") for chunk in top_chunks],
            "elapsed_seconds": round(total_elapsed_seconds, 3),
        }
        upsert_result(
            faithfulness_report["faithfulness_results"],
            {**common_ragas_result, "faithfulness_score": faithfulness_score},
        )
        upsert_result(
            correctness_report["answer_correctness_results"],
            {
                **common_ragas_result,
                "reference_answer": item["reference_answer"],
                "answer_correctness_score": answer_correctness_score,
            },
        )

        save_all_reports(reports)
        print(
            f"[{item['id']}] retrieval: "
            f"{'PASS' if retrieval_report['results'][-1]['passed'] else 'FAIL'} | "
            f"citation: {'PASS' if citation_report['citation_results'][-1]['citation_valid'] else 'FAIL'} | "
            f"faithfulness: {faithfulness_score if faithfulness_score is not None else 'not scored'} | "
            f"correctness: {answer_correctness_score if answer_correctness_score is not None else 'not scored'}"
        )

    if not stopped:
        for item in unanswerable_cases:
            if item["id"] in completed_refusal_ids:
                print(f"[SKIP] {item['id']} already evaluated")
                continue

            try:
                start_time = perf_counter()
                top_chunks = retrieve_chunks(item["question"])
                response = answer_from_chunks(item["question"], top_chunks)
                elapsed_seconds = perf_counter() - start_time
            except Exception as error:
                stopped = True
                for _, report in reports:
                    report["run_status"] = "paused_due_to_error"
                    report["last_error"] = str(error)
                save_all_reports(reports)
                print(f"[STOPPED] {item['id']}: {error}")
                break

            refusal_result = evaluate_refusal_case(
                item,
                response,
                elapsed_seconds,
            )
            upsert_result(refusal_report["refusal_results"], refusal_result)
            save_all_reports(reports)
            print(
                f"[{item['id']}] refusal: "
                f"{'PASS' if refusal_result['correct_refusal'] else 'FAIL'}"
            )

    update_retrieval_metrics(retrieval_report)
    update_citation_metrics(citation_report)
    update_refusal_metrics(refusal_report)
    update_ragas_metrics(
        faithfulness_report,
        "faithfulness_results",
        "faithfulness_score",
        "average_faithfulness",
    )
    update_ragas_metrics(
        correctness_report,
        "answer_correctness_results",
        "answer_correctness_score",
        "average_answer_correctness",
    )

    if not stopped:
        if len(retrieval_report["results"]) == len(answerable_cases):
            retrieval_report["run_status"] = "complete"
        if len(citation_report["citation_results"]) == len(answerable_cases):
            citation_report["run_status"] = "complete"
        if len(faithfulness_report["faithfulness_results"]) == len(answerable_cases):
            faithfulness_report["run_status"] = "complete"
        if len(correctness_report["answer_correctness_results"]) == len(answerable_cases):
            correctness_report["run_status"] = "complete"
        if len(refusal_report["refusal_results"]) == len(unanswerable_cases):
            refusal_report["run_status"] = "complete"

    save_all_reports(reports)


if __name__ == "__main__":
    run_evaluation()
