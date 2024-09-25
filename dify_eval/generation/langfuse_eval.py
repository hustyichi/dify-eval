import asyncio
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from generation.dify_chat import send_chat_message
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse()


def get_langfuse_dataset(dataset_name: str):
    return langfuse.get_dataset(dataset_name)


async def run_dataset_item(item, run_name, semaphore):
    async with semaphore:
        response = await send_chat_message(item.input)

        item.link(
            trace_or_observation=None,
            run_name=run_name,
            trace_id=response["message_id"],
            observation_id=None,
        )
        return response


def save_results(results: list[dict], local_path: str = ""):
    data = []
    for result in results:
        data.append(result["answer"].strip())

    df = pd.DataFrame(data, columns=["answer"])

    if not local_path:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = f'submit/results_{os.getenv("RUN_NAME", "")}_{current_time}.csv'

    df.to_csv(local_path, index=False)


async def run_dataset(
    dataset_name: str = os.getenv("DATASET_NAME", ""),
    run_name: str = os.getenv("RUN_NAME", ""),
    max_concurrency: int = 3,
):

    if not dataset_name:
        raise ValueError("No dataset name provided.")

    if not run_name:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dataset = get_langfuse_dataset(dataset_name)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    input_data = []
    for item in dataset.items:
        task = asyncio.create_task(run_dataset_item(item, run_name, semaphore))
        input_data.append(item.input)
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


async def run_dataset_and_save(
    dataset_name: str = os.getenv("DATASET_NAME", ""),
    run_name: str = os.getenv("RUN_NAME", ""),
    max_concurrency: int = 1,
):
    results = await run_dataset(dataset_name, run_name, max_concurrency)
    save_results(results)
