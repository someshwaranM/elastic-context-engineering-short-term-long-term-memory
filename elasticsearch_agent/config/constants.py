"""Constants and default values for Elasticsearch Agent."""

# Embedding model dimensions
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embedding_dimension(embedding_model: str) -> int:
    """
    Get embedding dimension for a given model.
    
    Args:
        embedding_model: Name of the embedding model
        
    Returns:
        Embedding dimension (default: 1536)
    """
    return EMBEDDING_DIMENSIONS.get(embedding_model, 1536)

