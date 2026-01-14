"""Relevance checking for Elasticsearch Agent."""

from typing import Tuple, Optional
import json
import re
from rich.console import Console


def check_context_relevance(
    query: str,
    context: str,
    llm,
    verbose: bool = False,
    console: Optional[Console] = None
) -> Tuple[bool, float]:
    """
    Use LLM to check if the retrieved context actually answers the user's query.
    
    Args:
        query: User's query
        context: Retrieved context from Elasticsearch
        llm: LLM instance for relevance checking
        verbose: Whether to show verbose output
        console: Rich console for output
        
    Returns:
        Tuple of (is_relevant, relevance_score)
        - is_relevant: True if context actually answers the query
        - relevance_score: Relevance score from LLM (0.0 to 1.0)
    """
    try:
        relevance_prompt = f"""You are an expert at evaluating whether retrieved context actually answers a user's question.

User's Question: "{query}"

Retrieved Context:
{context}

Evaluate whether the retrieved context actually answers the user's question. Consider:
1. Does the context contain information that directly answers the question?
2. Are there any key details missing (e.g., asking about "Bangalore" but context only mentions "Delhi")?
3. Is the context relevant to the specific question asked?

Respond with ONLY a JSON object in this exact format:
{{
    "is_relevant": true or false,
    "relevance_score": 0.0 to 1.0,
    "reason": "brief explanation"
}}

Be strict: if the question asks about a specific entity (like "Bangalore") but the context only mentions a different entity (like "Delhi"), it is NOT relevant, even if semantically similar."""

        response = llm.invoke(relevance_prompt).content
        
        # Parse JSON response
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*"is_relevant"[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            is_relevant = result.get("is_relevant", False)
            relevance_score = float(result.get("relevance_score", 0.0))
            reason = result.get("reason", "")
            
            if verbose and console:
                console.print(f"[cyan]🔍 Relevance check: {'✅ Relevant' if is_relevant else '❌ Not Relevant'} (score: {relevance_score:.2f})[/cyan]")
                if reason:
                    console.print(f"[cyan]   Reason: {reason}[/cyan]")
            
            return is_relevant, relevance_score
        else:
            # Fallback: try to parse the response more flexibly
            if "is_relevant" in response.lower() and "true" in response.lower():
                if verbose and console:
                    console.print(f"[yellow]⚠️  Could not parse relevance JSON, assuming relevant based on response[/yellow]")
                return True, 0.7
            else:
                if verbose and console:
                    console.print(f"[yellow]⚠️  Could not parse relevance JSON, assuming not relevant[/yellow]")
                return False, 0.3
                
    except Exception as e:
        if verbose and console:
            console.print(f"[yellow]⚠️  Error checking relevance: {e}[/yellow]")
            console.print(f"[yellow]⚠️  Assuming not relevant for safety[/yellow]")
        return False, 0.0

