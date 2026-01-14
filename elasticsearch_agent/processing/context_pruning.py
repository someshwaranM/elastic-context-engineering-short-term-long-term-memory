"""Context pruning with Provence reranker for Elasticsearch Agent."""

from typing import Optional, Any
from rich.console import Console


def prune_with_provence(
    query: str,
    context: str,
    provence_model: Optional[Any],
    threshold: float,
    verbose: bool = False,
    console: Optional[Console] = None
) -> str:
    """
    Prune context using Provence reranker model
    
    Args:
        query: User's query/question
        context: Original context to prune
        provence_model: Provence reranker model instance (None if unavailable)
        threshold: Relevance threshold (0-1) for Provence reranker.
                   0.1 = conservative (recommended, no performance drop)
                   0.3-0.5 = moderate to aggressive pruning
        verbose: Whether to show verbose output
        console: Rich console for output
        
    Returns:
        Pruned context with only relevant sentences
    """
    if provence_model is None:
        return context
    
    try:
        # Use Provence's process method
        provence_output = provence_model.process(
            question=query,
            context=context,
            threshold=threshold,
            always_select_title=False,
            enable_warnings=False
        )
        
        # Extract pruned context from output
        pruned_context = provence_output.get('pruned_context', context)
        reranking_score = provence_output.get('reranking_score', 0.0)
        
        # Log statistics
        original_length = len(context)
        pruned_length = len(pruned_context)
        reduction_pct = ((original_length - pruned_length) / original_length * 100) if original_length > 0 else 0
        
        if verbose and console:
            console.print(f"[cyan]📊 Pruning stats: {pruned_length}/{original_length} chars ({reduction_pct:.1f}% reduction, threshold={threshold:.2f}, rerank_score={reranking_score:.3f})[/cyan]")
        
        return pruned_context if pruned_context else context
        
    except Exception as e:
        if console:
            console.print(f"[yellow]⚠️ Error in Provence pruning: {str(e)}[/yellow]")
            console.print(f"[yellow]⚠️ Falling back to original context[/yellow]")
        return context

