"""Provence reranker initialization for Elasticsearch Agent."""

from typing import Optional, Any
from rich.console import Console


def initialize_provence(args, console: Console) -> Optional[Any]:
    """
    Initialize Provence reranker model (optional, for context pruning).
    
    Args:
        args: Parsed command line arguments
        console: Rich console for output
        
    Returns:
        Provence model instance or None if unavailable
    """
    try:
        from transformers import AutoModel
        import nltk
        
        # Download nltk data if needed
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            nltk.download('punkt', quiet=True)
        
        console.print("[yellow]🔧 Loading Provence reranker model...[/yellow]")
        provence_model = AutoModel.from_pretrained(
            "naver/provence-reranker-debertav3-v1",
            trust_remote_code=True
        )
        provence_model.eval()
        console.print(f"[green]✅ Provence reranker loaded successfully![/green]")
        console.print(f"[blue]📊 Pruning threshold: {args.pruning_threshold} (0.1=conservative, 0.5=aggressive)[/blue]")
        return provence_model
    except Exception as e:
        console.print(f"[yellow]⚠️  Provence reranker not available: {e}[/yellow]")
        console.print("[yellow]⚠️  Continuing without context pruning...[/yellow]")
        return None

