"""Configuration module for Elasticsearch Agent."""

from .settings import get_args
from .constants import EMBEDDING_DIMENSIONS, get_embedding_dimension

__all__ = ["get_args", "EMBEDDING_DIMENSIONS", "get_embedding_dimension"]

