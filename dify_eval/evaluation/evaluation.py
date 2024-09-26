import os

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.client import FetchTracesResponse

from dify_eval.evaluation import constants

load_dotenv()

langfuse = Langfuse()


def get_run_traces(
    user_id: str = os.getenv("RUN_NAME", "auto_test_user"),
    page: int = 0,
    limit: int = constants.BATCH_SIZE,
) -> FetchTracesResponse:
    return langfuse.fetch_traces(user_id=user_id, page=page, limit=limit)
