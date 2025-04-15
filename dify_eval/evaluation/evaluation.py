import json
import os
from typing import Optional

from datasets import Dataset
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.client import ObservationsView, TraceWithDetails
from loguru import logger
from ragas import evaluate
from ragas.metrics.base import Metric as RagasMetric

from dify_eval.evaluation import constants, ragas_models
from dify_eval.evaluation.metrics import retrieval_evaluate

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

    logger.info(
        f" >> Finish trace {trace_id} with question {dataset_dict['question']} evaluation result: {result}"
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
    metrics,
    trace_data: dict,
    ground_truth_map: dict = {},
):
    QUERY_KEY = "sys.query"
    ANSWER_KEY = "answer"

    logger.info(
        f" >> Start evaluate trace {trace_data['id']} with {trace_data['input'].get(QUERY_KEY, trace_data['input'])}"
    )
    knowledge_retrieval_observations = get_knowledge_retrieval_observations(
        trace_data["id"]
    )
    logger.debug(
        f"Trace {trace_data['id']} with {trace_data['input'].get(QUERY_KEY, trace_data['input'])} got {len(knowledge_retrieval_observations)} knowledge retrievals"
    )
    if not knowledge_retrieval_observations:
        logger.warning(
            f"Trace {trace_data['id']} with {trace_data['input'].get(QUERY_KEY, trace_data['input'])} has no knowledge retrievals, skip evaluation"
        )
        return

    # get the last knowledge retrieval content
    trace_knowlege_retrieval_content = get_knowledge_retrieval_content(
        knowledge_retrieval_observations[-1]
    )
    logger.debug(
        f"Trace {trace_data['id']} got {len(trace_knowlege_retrieval_content)} contexts"
    )

    data_sample = {
        "question": [trace_data["input"].get(QUERY_KEY, trace_data["input"])],
        "answer": [trace_data["output"].get(ANSWER_KEY, trace_data["output"])],
        "contexts": [trace_knowlege_retrieval_content],
        "ground_truth": [
            ground_truth_map.get(
                trace_data["input"].get(QUERY_KEY, trace_data["input"]), ""
            )
        ],
    }

    retrieval_metrics = [
        m for m in metrics if isinstance(m, str) and m in constants.RETRIEVAL_METRICS
    ]
    ragas_metrics = [m for m in metrics if isinstance(m, RagasMetric)]
    try:
        if ragas_metrics:
            raw_ragas_evaluate(data_sample, ragas_metrics, trace_data["id"])
        if retrieval_metrics:
            retrieval_evaluate(data_sample, retrieval_metrics, trace_data["id"])
    except Exception as e:
        logger.exception(
            f"Trace {trace_data['id']} with {trace_data['input'].get(QUERY_KEY, trace_data['input'])} evaluate got error: {e}"
        )


def do_evaluate(
    metrics: list,
    run_name: str = os.getenv("RUN_NAME", "auto_test_user"),
    page: int = 1,
    limit: int = constants.BATCH_SIZE,
    ground_truth_map: dict = {},
):
    traces = get_run_traces(user_id=run_name, page=page, limit=limit)

    logger.info(
        f"Page {page} with limit {limit}, {len(traces)} traces found, start evaluating..."
    )

    for trace in traces:
        parsed_input = (
            json.loads(trace.input) if isinstance(trace.input, str) else trace.input
        )
        parsed_output = (
            json.loads(trace.output) if isinstance(trace.output, str) else trace.output
        )
        trace_data = {"id": trace.id, "input": parsed_input, "output": parsed_output}
        do_trace_evaluate(metrics, trace_data, ground_truth_map)

    return len(traces)


def get_ground_truth_map(dataset_name: str = os.getenv("DATASET_NAME", "")):
    dataset = langfuse.get_dataset(dataset_name)
    ground_truth_map = {}

    for item in dataset.items:
        if not (item.expected_output and item.input):
            continue

        ground_truth_map[item.input] = item.expected_output

    return ground_truth_map


def evaluate_dataset_run_items(
    metrics: list,
    run_name: str = os.getenv("RUN_NAME", "auto_test_user"),
    dataset_name: str = os.getenv("DATASET_NAME", ""),
):
    ground_truth_map = get_ground_truth_map(dataset_name)
    logger.debug(f"Evaluate {dataset_name} got {len(ground_truth_map)} expected output")

    page = 1
    while True:
        count = do_evaluate(
            metrics, run_name, page, constants.BATCH_SIZE, ground_truth_map
        )
        page += 1
        if count < constants.BATCH_SIZE:
            break

    logger.info(f"Finish evaluate dataset {dataset_name}")
