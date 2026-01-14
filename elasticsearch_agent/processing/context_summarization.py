"""Context summarization for Elasticsearch Agent."""

from typing import Optional
from rich.console import Console


def summarize_context(
    query: str,
    context: str,
    llm,
    verbose: bool = False,
    console: Optional[Console] = None
) -> str:
    """
    Summarize context using LLM to reduce duplication and focus on relevant information
    
    Args:
        query: User's query/question
        context: Context to summarize
        llm: LLM instance for summarization
        verbose: Whether to show verbose output
        console: Rich console for output
        
    Returns:
        Summarized context
    """
    try:
        summary_prompt = f"""You are an expert at summarizing conversation context.

Your task: Analyze the provided conversation context and produce a condensed summary that fully answers or supports the user's specific question.

The summary must:
1. Preserve every fact, detail, and information that directly relates to the question
2. Eliminate redundancy and duplicate information
3. Maintain chronological flow when relevant
4. Focus on information that helps answer: "{query}"

Context to summarize:
{context}

Provide a concise summary that preserves all relevant information:"""

        summary = llm.invoke(summary_prompt).content
        
        if verbose and console:
            original_length = len(context)
            summary_length = len(summary)
            reduction_pct = ((original_length - summary_length) / original_length * 100) if original_length > 0 else 0
            console.print(f"[cyan]📝 Summarization stats: {summary_length}/{original_length} chars ({reduction_pct:.1f}% reduction)[/cyan]")
        
        return summary
        
    except Exception as e:
        if console:
            console.print(f"[yellow]⚠️ Error in context summarization: {str(e)}[/yellow]")
            console.print(f"[yellow]⚠️ Falling back to original context[/yellow]")
        return context

