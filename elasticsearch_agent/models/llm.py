"""LLM initialization for Elasticsearch Agent."""

import os
from langchain_openai import ChatOpenAI


def initialize_llm() -> ChatOpenAI:
    """
    Initialize OpenAI LLM.
    
    Returns:
        Initialized ChatOpenAI instance
    """
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

