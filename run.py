import asyncio

from dify_eval.dataset import dataset
from dify_eval.generation import generation

if __name__ == "__main__":
    # upload local file to langfuse dataset
    dataset.create_dataset_from_file(metadata_column="ques_id")

    # run dify call by dataset
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(generation.run_dataset_generation())
