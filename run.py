import asyncio

from dify_eval.eval import langfuse_eval

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(langfuse_eval.run_dataset_and_save())
    print(results)
