import os

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


def get_ragas_llm_and_embeddings():
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("RAGAS_EVAL_LLM"),
            openai_api_base=os.getenv("RAGAS_BASE_URL"),
            openai_api_key=os.getenv("RAGAS_API_KEY"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=os.getenv("RAGAS_EMBEDDING"),
            base_url=os.getenv("RAGAS_BASE_URL"),
            api_key=os.getenv("RAGAS_API_KEY"),
        )
    )
    return llm, embeddings
