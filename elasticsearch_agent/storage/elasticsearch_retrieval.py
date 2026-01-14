"""Elasticsearch retrieval functions for Elasticsearch Agent."""

from typing import List, Dict, Any, Tuple, Optional
from elasticsearch import Elasticsearch
from rich.console import Console
from ..processing.context_pruning import prune_with_provence
from ..processing.context_summarization import summarize_context
from ..processing.relevance_check import check_context_relevance


def retrieve_from_elasticsearch(
    query: str,
    es_client: Optional[Elasticsearch],
    es_index_name: Optional[str],
    embeddings,
    rank_window: int,
    k: int = 5,
    verbose: bool = False,
    console: Optional[Console] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Retrieve context from Elasticsearch with score-based ranking
    
    Args:
        query: Search query
        es_client: Elasticsearch client instance
        es_index_name: Name of the Elasticsearch index
        embeddings: Embeddings model instance
        rank_window: Number of candidates to retrieve before ranking
        k: Number of results to return
        verbose: Whether to show verbose output
        console: Rich console for output
        
    Returns:
        Tuple of (retrieved_documents, formatted_context_string)
    """
    if not es_client or not es_index_name:
        return [], "Elasticsearch is not available. Cannot search long-term memory."
    
    try:
        # Check if index exists and has documents
        if not es_client.indices.exists(index=es_index_name):
            return [], "No previous conversations stored in long-term memory yet."
        
        # Get document count
        try:
            doc_count = es_client.count(index=es_index_name)["count"]
            if doc_count == 0:
                return [], "Long-term memory is empty. No previous conversations to search."
        except Exception as e:
            return [], f"Error checking memory: {str(e)}"
        
        # Generate embedding for the query
        try:
            query_embedding = embeddings.embed_query(query)
        except Exception as e:
            return [], f"Error generating embedding: {str(e)}"
        
        # Perform semantic search using kNN with rank_window
        try:
            search_body = {
                "knn": {
                    "field": "vector",
                    "query_vector": query_embedding,
                    "k": k,
                    "num_candidates": rank_window  # Retrieve more candidates, then rank top k
                },
                "_source": ["text", "content", "message_type", "timestamp", "thread_id"],
                "size": k
            }
            
            response = es_client.search(index=es_index_name, body=search_body)
            
            if not response.get("hits") or len(response["hits"]["hits"]) == 0:
                return [], "No relevant previous conversations found in long-term memory."
            
            # Extract documents with scores
            retrieved_docs = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                score = hit["_score"]
                retrieved_docs.append({
                    "content": source.get("content", source.get("text", "")),
                    "message_type": source.get("message_type", "unknown"),
                    "timestamp": source.get("timestamp", "unknown"),
                    "thread_id": source.get("thread_id", "unknown"),
                    "score": score
                })
            
            # Format context string
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(doc["content"])
            
            context_string = "\n\n".join(context_parts)
            
            # Verbose display
            if verbose and console:
                console.print(f"\n[bold yellow]🔍 RETRIEVAL ANALYSIS[/bold yellow]")
                console.print("="*80)
                console.print(f"[blue]Query:[/blue] {query}")
                console.print(f"[blue]Retrieved:[/blue] {len(retrieved_docs)} documents (from {rank_window} candidates)")
                console.print(f"[blue]Total context length:[/blue] {len(context_string)} characters\n")
                
                for i, doc in enumerate(retrieved_docs, 1):
                    console.print(f"[cyan]📄 Document {i} | Score: {doc['score']:.4f} | Type: {doc['message_type']}[/cyan]")
                    console.print(f"[cyan]   Timestamp: {doc['timestamp']} | Thread: {doc['thread_id']}[/cyan]")
                    content_preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                    console.print(f"[cyan]   Content: {content_preview}[/cyan]")
                    console.print("-" * 80)
            
            return retrieved_docs, context_string
            
        except Exception as e:
            return [], f"Error searching memory: {str(e)}"
            
    except Exception as e:
        return [], f"Error accessing long-term memory: {str(e)}"


def check_elasticsearch_with_confidence(
    query: str,
    es_client: Optional[Elasticsearch],
    es_index_name: Optional[str],
    embeddings,
    llm,
    provence_model: Optional[Any],
    args,
    console: Optional[Console] = None
) -> Tuple[bool, str, float]:
    """
    Check Elasticsearch for relevant context and return confidence score.
    Now includes LLM-based relevance checking to ensure context actually answers the query.
    
    Args:
        query: User's query
        es_client: Elasticsearch client instance
        es_index_name: Name of the Elasticsearch index
        embeddings: Embeddings model instance
        llm: LLM instance for relevance checking
        provence_model: Provence reranker model (optional)
        args: Parsed command line arguments
        console: Rich console for output
        
    Returns:
        Tuple of (has_results, context_or_message, max_score)
        - has_results: True if Elasticsearch has relevant results that actually answer the query
        - context_or_message: Pruned and summarized context, or error message
        - max_score: Maximum similarity score from Elasticsearch (0.0 if no results)
    """
    if not es_client or not es_index_name:
        return False, "Elasticsearch is not available.", 0.0
    
    try:
        # Retrieve context from Elasticsearch
        retrieved_docs, context_string = retrieve_from_elasticsearch(
            query, es_client, es_index_name, embeddings, args.rank_window,
            k=5, verbose=args.verbose, console=console
        )
        
        if not retrieved_docs:
            return False, context_string, 0.0  # No results or error
        
        # Get maximum score from retrieved documents
        max_score = max(doc["score"] for doc in retrieved_docs) if retrieved_docs else 0.0
        
        # Check similarity score threshold first
        if max_score < args.confidence_threshold:
            if args.verbose and console:
                console.print(f"[yellow]⚠️  Elasticsearch similarity score ({max_score:.4f}) below threshold ({args.confidence_threshold}). Will use Tavily.[/yellow]")
            return False, f"Elasticsearch similarity score ({max_score:.4f}) below threshold ({args.confidence_threshold})", max_score
        
        # Similarity score is good, now process context
        original_context = context_string
        
        if args.verbose and console:
            console.print(f"[green]✅ Elasticsearch similarity score ({max_score:.4f}) meets threshold ({args.confidence_threshold})[/green]")
            console.print(f"[yellow]📦 Original retrieved context: {len(original_context)} characters[/yellow]")
        
        # Step 1: Prune context using Provence (if available)
        if provence_model:
            if args.verbose and console:
                console.print(f"[yellow]📝 Pruning context with Provence reranker...[/yellow]")
            pruned_context = prune_with_provence(query, original_context, provence_model, args.pruning_threshold, args.verbose, console)
        else:
            pruned_context = original_context
        
        # Step 2: Summarize context to reduce duplication
        if args.verbose and console:
            console.print(f"[yellow]📝 Summarizing context to reduce duplication...[/yellow]")
        summarized_context = summarize_context(query, pruned_context, llm, args.verbose, console)
        
        # Step 3: Check actual relevance using LLM
        if args.verbose and console:
            console.print(f"[yellow]🔍 Checking if context actually answers the question...[/yellow]")
        
        is_relevant, relevance_score = check_context_relevance(query, summarized_context, llm, args.verbose, console)
        
        if not is_relevant:
            if args.verbose and console:
                console.print(f"[yellow]⚠️  Retrieved context is not actually relevant to the question (relevance score: {relevance_score:.2f}). Will use Tavily.[/yellow]")
            return False, f"Retrieved context is not relevant to the question (relevance score: {relevance_score:.2f})", max_score
        
        # Both similarity and relevance checks passed
        if args.verbose and console:
            console.print(f"[green]✅ Context is relevant (relevance score: {relevance_score:.2f})[/green]")
        
        # Format final result
        result = f"Found {len(retrieved_docs)} relevant previous conversation(s) (similarity: {max_score:.4f}, relevance: {relevance_score:.2f}):\n\n"
        result += f"Context Summary:\n{summarized_context}"
        
        if args.verbose and console:
            console.print(f"[green]✅ Final context ready: {len(summarized_context)} characters[/green]")
        
        return True, result, max_score
        
    except Exception as e:
        if args.verbose and console:
            console.print(f"[red]❌ Error checking Elasticsearch: {e}[/red]")
        return False, f"Error accessing long-term memory: {str(e)}", 0.0

