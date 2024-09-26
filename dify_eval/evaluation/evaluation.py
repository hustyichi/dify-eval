import os

from datasets import Dataset
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.client import FetchTracesResponse, TraceWithDetails
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
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
    page: int = 0,
    limit: int = constants.BATCH_SIZE,
) -> FetchTracesResponse:
    return langfuse.fetch_traces(user_id=user_id, page=page, limit=limit)


def do_trace_evaluate(
    trace: TraceWithDetails,
    llm: LangchainLLMWrapper,
    embedding_model: LangchainEmbeddingsWrapper,
):

    # TODO: get context from trace
    data_sample = {
        "question": [trace.input],
        "answer": [trace.output],
        "contexts": [],
        "ground_truth": [],
    }
    dataset = Dataset.from_dict(data_sample)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_recall,
            context_precision,
        ],
        llm=llm,
        embeddings=embedding_model,
    )

    for metric_type, metric_score in result.items():
        langfuse.score(
            trace_id=trace.id,
            name=metric_type,
            value=metric_score,
        )


def do_evaluate(
    user_id: str = os.getenv("RUN_NAME", "auto_test_user"),
    page: int = 0,
    limit: int = constants.BATCH_SIZE,
    llm: LangchainLLMWrapper = None,
    embedding_model: LangchainEmbeddingsWrapper = None,
):
    traces = get_run_traces(user_id=user_id, page=page, limit=limit)

    for idx in range(len(traces)):
        do_trace_evaluate(traces[idx], llm, embedding_model)

    return len(traces)


def evaluate_dataset_run_items(user_id: str = os.getenv("RUN_NAME", "auto_test_user")):
    llm, embedding = ragas_models.get_ragas_llm_and_embeddings()
    page = 0
    while True:
        count = do_evaluate(user_id, page, constants.BATCH_SIZE, llm, embedding)
        page += 1
        if count < constants.BATCH_SIZE:
            break
