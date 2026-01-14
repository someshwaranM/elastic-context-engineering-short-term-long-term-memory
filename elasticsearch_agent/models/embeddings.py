"""Embeddings initialization for Elasticsearch Agent."""

import os
from langchain_openai import OpenAIEmbeddings
from ..config.constants import get_embedding_dimension


def initialize_embeddings():
    """
    Initialize OpenAI Embeddings.
    
    Returns:
        Tuple of (embeddings, embedding_model, embedding_dimension)
    """
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    embedding_dimension = get_embedding_dimension(embedding_model)
    
    return embeddings, embedding_model, embedding_dimension

