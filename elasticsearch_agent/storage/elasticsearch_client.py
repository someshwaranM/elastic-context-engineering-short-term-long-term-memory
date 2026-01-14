"""Elasticsearch client initialization for Elasticsearch Agent."""

import os
from typing import Tuple, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError, ConnectionError as ESConnectionError
from rich.console import Console


def initialize_elasticsearch(embedding_dimension: int, console: Console) -> Tuple[Optional[Elasticsearch], Optional[str], bool]:
    """
    Initialize Elasticsearch client and create index if needed.
    
    Args:
        embedding_dimension: Dimension of embeddings (for index naming)
        console: Rich console for output
        
    Returns:
        Tuple of (es_client, index_name, is_new_index)
        Returns (None, None, False) if connection fails
    """
    es_url = os.getenv("ELASTICSEARCH_URL")
    es_api_key = os.getenv("ELASTICSEARCH_API_KEY")
    
    if not es_url or not es_api_key:
        console.print("[yellow]⚠️  ELASTICSEARCH_URL and ELASTICSEARCH_API_KEY not provided[/yellow]")
        return None, None, False
    
    try:
        es_client = Elasticsearch(
            hosts=[es_url],
            api_key=es_api_key,
            request_timeout=30
        )
        
        # Test connection with proper exception handling
        if not es_client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch - ping failed")
        
        # Index name
        index_name = f"langgraph-agent-memories-{embedding_dimension}"
        
        # Check if index exists
        index_exists = False
        try:
            index_exists = es_client.indices.exists(index=index_name)
        except Exception as e:
            console.print(f"[red]❌ Error checking index existence: {e}[/red]")
            raise
        
        is_new_index = False
        
        if not index_exists:
            # This is the first time - create the index
            console.print(f"[yellow]🔧 Creating new Elasticsearch index: {index_name}...[/yellow]")
            is_new_index = True
            
            try:
                index_mapping = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "vector": {
                                "type": "dense_vector",
                                "dims": embedding_dimension,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "thread_id": {"type": "keyword"},
                            "checkpoint_id": {"type": "keyword"},
                            "timestamp": {"type": "date"},
                            "message_type": {"type": "keyword"},
                            "message_id": {"type": "keyword"},
                            "content": {"type": "text"}
                        }
                    }
                }
                es_client.indices.create(index=index_name, body=index_mapping)
                console.print(f"[green]✅ Elasticsearch index created successfully![/green]")
            except RequestError as e:
                if "resource_already_exists_exception" in str(e):
                    # Index was created between check and create (race condition)
                    console.print(f"[yellow]⚠️  Index was created by another process, using existing index[/yellow]")
                    is_new_index = False
                else:
                    console.print(f"[red]❌ Error creating index: {e}[/red]")
                    raise
            except Exception as e:
                console.print(f"[red]❌ Unexpected error creating index: {e}[/red]")
                raise
        else:
            # Index already exists
            console.print(f"[green]📋 Using existing Elasticsearch index: {index_name}[/green]")
            
            # Verify index has the correct mapping
            try:
                current_mapping = es_client.indices.get_mapping(index=index_name)
                # Check if vector field exists with correct dimension
                props = current_mapping[index_name]["mappings"].get("properties", {})
                if "vector" not in props:
                    console.print(f"[yellow]⚠️  Warning: Index exists but missing vector field. May need to recreate.[/yellow]")
                elif props.get("vector", {}).get("dims") != embedding_dimension:
                    console.print(f"[yellow]⚠️  Warning: Index vector dimension mismatch. Expected {embedding_dimension}, found {props.get('vector', {}).get('dims')}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not verify index mapping: {e}[/yellow]")
        
        return es_client, index_name, is_new_index
        
    except ESConnectionError as e:
        console.print(f"[red]❌ Elasticsearch connection error: {e}[/red]")
        return None, None, False
    except Exception as e:
        console.print(f"[red]❌ Error initializing Elasticsearch: {e}[/red]")
        return None, None, False

