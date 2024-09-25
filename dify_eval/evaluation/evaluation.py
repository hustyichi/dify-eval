import os

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse()


def get_run_traces(user_id: str = os.getenv("RUN_NAME", "auto_test_user")):
    return langfuse.fetch_traces(user_id=user_id)
