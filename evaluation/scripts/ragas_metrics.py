import os
from time import perf_counter

import instructor
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerCorrectness, Faithfulness


def create_ragas_metrics():
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_client = AsyncOpenAI(
        base_url=f"{ollama_base_url.rstrip('/')}/v1",
        api_key="ollama",
    )
    evaluator_llm = llm_factory(
        os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        client=ollama_client,
        mode=instructor.Mode.MD_JSON,
    )
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    return (
        Faithfulness(llm=evaluator_llm),
        AnswerCorrectness(llm=evaluator_llm, embeddings=embeddings),
    )


def evaluate_ragas_case(item, response, top_chunks, faithfulness_metric, correctness_metric):
    common_result = {
        "id": item["id"],
        "question": item["question"],
        "actual_status": response.get("status"),
        "retrieved_chunk_ids": [chunk.get("chunk_id") for chunk in top_chunks],
    }

    if response.get("status") != "answered":
        return (
            {**common_result, "faithfulness_score": None, "elapsed_seconds": 0},
            {
                **common_result,
                "reference_answer": item["reference_answer"],
                "answer_correctness_score": None,
                "elapsed_seconds": 0,
            },
        )

    contexts = [chunk["page_content"] for chunk in top_chunks]

    start_time = perf_counter()
    faithfulness_score = float(
        faithfulness_metric.score(
            user_input=item["question"],
            response=response["response"],
            retrieved_contexts=contexts,
        ).value
    )
    faithfulness_elapsed_seconds = perf_counter() - start_time

    start_time = perf_counter()
    answer_correctness_score = float(
        correctness_metric.score(
            user_input=item["question"],
            response=response["response"],
            reference=item["reference_answer"],
        ).value
    )
    correctness_elapsed_seconds = perf_counter() - start_time

    return (
        {
            **common_result,
            "faithfulness_score": faithfulness_score,
            "elapsed_seconds": round(faithfulness_elapsed_seconds, 3),
        },
        {
            **common_result,
            "reference_answer": item["reference_answer"],
            "answer_correctness_score": answer_correctness_score,
            "elapsed_seconds": round(correctness_elapsed_seconds, 3),
        },
    )


def update_ragas_metrics(results, score_key, metric_name):
    scored_results = [result for result in results if result[score_key] is not None]
    total = len(results)

    return {
        metric_name: round(
            sum(result[score_key] for result in scored_results) / len(scored_results),
            3,
        ) if scored_results else 0,
        "scored_cases": len(scored_results),
        "evaluated_cases": total,
        "average_elapsed_seconds": round(
            sum(result["elapsed_seconds"] for result in results) / total,
            3,
        ) if total else 0,
    }
