"""Tool definitions for Elasticsearch Agent."""

from typing import Optional, Any
from elasticsearch import Elasticsearch
from langchain_core.tools import StructuredTool
from rich.console import Console
from ..storage.elasticsearch_retrieval import retrieve_from_elasticsearch
from ..processing.context_pruning import prune_with_provence
from ..processing.context_summarization import summarize_context


def search_elasticsearch_memory(
    query: str,
    es_client: Optional[Elasticsearch],
    es_index_name: Optional[str],
    embeddings,
    llm,
    provence_model: Optional[Any],
    args,
    console: Optional[Console] = None
) -> str:
    """
    Search Elasticsearch for relevant previous conversations, prune context, summarize, and return.
    This tool searches long-term memory before using other tools.
    
    Args:
        query: The search query to find relevant past conversations
        es_client: Elasticsearch client instance
        es_index_name: Name of the Elasticsearch index
        embeddings: Embeddings model instance
        llm: LLM instance for summarization
        provence_model: Provence reranker model (optional)
        args: Parsed command line arguments
        console: Rich console for output
        
    Returns:
        A string containing relevant past conversation context (pruned and summarized)
    """
    if not es_client or not es_index_name:
        return "Elasticsearch is not available. Cannot search long-term memory."
    
    try:
        # Retrieve context from Elasticsearch (use rank_window for candidates, return top 5)
        retrieved_docs, context_string = retrieve_from_elasticsearch(
            query, es_client, es_index_name, embeddings, args.rank_window,
            k=5, verbose=args.verbose, console=console
        )
        
        if not retrieved_docs:
            return context_string  # This will be an error message or empty message
        
        # Combine all retrieved context
        original_context = context_string
        
        if args.verbose and console:
            console.print(f"\n[yellow]📦 Original retrieved context: {len(original_context)} characters[/yellow]")
        
        # Step 1: Prune context using Provence (if available)
        if provence_model:
            if args.verbose and console:
                console.print(f"[yellow]📝 Pruning context with Provence reranker...[/yellow]")
            pruned_context = prune_with_provence(
                query, original_context, provence_model, args.pruning_threshold,
                args.verbose, console
            )
        else:
            pruned_context = original_context
        
        # Step 2: Summarize context to reduce duplication
        if args.verbose and console:
            console.print(f"[yellow]📝 Summarizing context to reduce duplication...[/yellow]")
        summarized_context = summarize_context(query, pruned_context, llm, args.verbose, console)
        
        # Format final result
        result = f"Found {len(retrieved_docs)} relevant previous conversation(s) (retrieved from rank_window={args.rank_window} candidates):\n\n"
        result += f"Context Summary:\n{summarized_context}"
        
        if args.verbose and console:
            console.print(f"[green]✅ Final context ready: {len(summarized_context)} characters[/green]")
        
        return result
        
    except Exception as e:
        return f"Error accessing long-term memory: {str(e)}"


def create_elasticsearch_memory_tool(
    es_client: Optional[Elasticsearch],
    es_index_name: Optional[str],
    embeddings,
    llm,
    provence_model: Optional[Any],
    args,
    console: Optional[Console] = None
) -> Optional[StructuredTool]:
    """
    Create the Elasticsearch memory tool.
    
    Args:
        es_client: Elasticsearch client instance
        es_index_name: Name of the Elasticsearch index
        embeddings: Embeddings model instance
        llm: LLM instance
        provence_model: Provence reranker model (optional)
        args: Parsed command line arguments
        console: Rich console for output
        
    Returns:
        StructuredTool instance or None if Elasticsearch unavailable
    """
    if not es_client or not es_index_name:
        return None
    
    # Create a closure to capture dependencies
    def tool_func(query: str) -> str:
        return search_elasticsearch_memory(
            query, es_client, es_index_name, embeddings, llm, provence_model, args, console
        )
    
    return StructuredTool.from_function(
        func=tool_func,
        name="search_long_term_memory",
        description="""Search long-term memory for previous conversations and context. 
        Use this tool FIRST when you need to recall information from past conversations, 
        such as user names, preferences, or previous topics discussed. 
        This searches stored conversation history, prunes irrelevant content, and summarizes to avoid duplication before using web search.""",
    )

