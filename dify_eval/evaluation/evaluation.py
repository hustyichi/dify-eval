import os
from typing import Optional

from datasets import Dataset
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.client import ObservationsView, TraceWithDetails
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from dify_eval.evaluation import constants, ragas_models

load_dotenv()

langfuse = Langfuse()


def get_run_traces(
    user_id: str = os.getenv("RUN_NAME", "auto_test_user"),
    page: int = 1,
    limit: int = constants.BATCH_SIZE,
) -> list[TraceWithDetails]:
    return langfuse.fetch_traces(user_id=user_id, page=page, limit=limit).data


def get_trace_observations(trace_id: str) -> list[ObservationsView]:
    return langfuse.fetch_observations(trace_id=trace_id).data


def identify_knowledge_retrieval(observation: ObservationsView) -> bool:
    KNOWLEDGE_RETRIEVAL_NAME = "knowledge-retrieval"

    if observation.name == KNOWLEDGE_RETRIEVAL_NAME:
        return True
    else:
        return False


def get_knowledge_retrieval_observations(trace_id: str) -> list[ObservationsView]:
    observations = get_trace_observations(trace_id)
    return [obs for obs in observations if identify_knowledge_retrieval(obs)]


def get_knowledge_retrieval_content(observation: ObservationsView) -> list[str]:
    RESULT_KEY = "result"

    result = observation.output.get(RESULT_KEY, [])
    return [item["content"] for item in result if item.get("content")]


def raw_ragas_evaluate(
    dataset_dict: dict, metrics: list, trace_id: Optional[str] = None
):
    llm, embedding_model = ragas_models.get_ragas_llm_and_embeddings()
    dataset = Dataset.from_dict(dataset_dict)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embedding_model,
    )

    logger.debug(
        f"Trace {trace_id} with question {dataset_dict['question']} evaluation result: {result}"
    )

    if trace_id:
        for metric_type, metric_score in result.items():
            langfuse.score(
                trace_id=trace_id,
                name=metric_type,
                value=metric_score,
            )

    return result


def do_trace_evaluate(
    trace: TraceWithDetails,
):
    QUERY_KEY = "sys.query"
    ANSWER_KEY = "answer"

    knowledge_retrieval_observations = get_knowledge_retrieval_observations(trace.id)
    logger.info(
        f"Trace {trace.id} got {len(knowledge_retrieval_observations)} knowledge retrievals"
    )
    if not knowledge_retrieval_observations:
        logger.warning(
            f"Trace {trace.id} with {trace.input.get(QUERY_KEY, trace.input)} has no knowledge retrievals, skip evaluation"
        )
        return

    # get the last knowledge retrieval content
    trace_knowlege_retrieval_content = get_knowledge_retrieval_content(
        knowledge_retrieval_observations[-1]
    )
    logger.info(
        f"Trace {trace.id} got {len(trace_knowlege_retrieval_content)} contexts"
    )

    data_sample = {
        "question": [trace.input.get(QUERY_KEY, trace.input)],
        "answer": [trace.output.get(ANSWER_KEY, trace.output)],
        "contexts": [trace_knowlege_retrieval_content],
        "ground_truth": [],
    }
    metrics = [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_recall,
        context_precision,
    ]

    raw_ragas_evaluate(data_sample, metrics, trace.id)


def do_evaluate(
    run_name: str = os.getenv("RUN_NAME", "auto_test_user"),
    page: int = 1,
    limit: int = constants.BATCH_SIZE,
):
    traces = get_run_traces(user_id=run_name, page=page, limit=limit)

    logger.info(f"Current {page} page, {len(traces)} traces found, start evaluating...")

    for idx in range(len(traces)):
        do_trace_evaluate(traces[idx])

    return len(traces)


def evaluate_dataset_run_items(run_name: str = os.getenv("RUN_NAME", "auto_test_user")):
    page = 1
    while True:
        count = do_evaluate(run_name, page, constants.BATCH_SIZE)
        page += 1
        if count < constants.BATCH_SIZE:
            break
