import os
from typing import Literal

import aiohttp
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()


@retry(stop=stop_after_attempt(10), wait=wait_fixed(3))
async def send_chat_message(
    query: str,
    inputs: dict = {},
    url: str = os.getenv("DIFY_API_BASE", ""),
    api_key: str = os.getenv("DIFY_API_KEY", ""),
    response_mode: Literal["streaming", "blocking"] = "blocking",
    user: str = os.getenv("RUN_NAME", "auto_test_user"),
    file_array: list = [],
):
    if not (url and api_key):
        raise ValueError(
            "DIFY_API_BASE and DIFY_API_KEY must be set in the environment variables"
        )

    chat_url = f"{url}/chat-messages"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": inputs,
        "query": query,
        "response_mode": response_mode,
        "conversation_id": "",
        "user": user,
        "files": file_array,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(chat_url, headers=headers, json=payload) as response:
            ret = await response.json()
            status = ret.get("status")
            message = ret.get("message")
            if status and message:
                raise ValueError(f"{status}: {message}")
            if ret.get("answer") == "":
                raise ValueError("Empty answer")
            return ret
