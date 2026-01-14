"""Agent factory for creating LangGraph agents."""

from typing import List, Optional, Any, Tuple
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from elasticsearch import Elasticsearch
from rich.console import Console
from .tools import create_elasticsearch_memory_tool


def create_agent(
    llm: ChatOpenAI,
    es_client: Optional[Elasticsearch],
    es_index_name: Optional[str],
    embeddings,
    provence_model: Optional[Any],
    args,
    console: Console
) -> Tuple[Any, MemorySaver, List[Any]]:
    """
    Create a LangGraph agent with appropriate tools.
    
    Args:
        llm: ChatOpenAI instance
        es_client: Elasticsearch client instance (optional)
        es_index_name: Name of the Elasticsearch index (optional)
        embeddings: Embeddings model instance
        provence_model: Provence reranker model (optional)
        args: Parsed command line arguments
        console: Rich console for output
        
    Returns:
        Tuple of (agent, memory_checkpointer, tools_list)
    """
    tools: List[Any] = []
    
    # Add Elasticsearch long-term memory tool if available
    elasticsearch_tool = create_elasticsearch_memory_tool(
        es_client, es_index_name, embeddings, llm, provence_model, args, console
    )
    
    if elasticsearch_tool:
        tools.append(elasticsearch_tool)
        console.print("[green]🧠 Elasticsearch long-term memory enabled - agent will search previous conversations first[/green]")
        if provence_model:
            console.print("[green]📝 Provence reranker enabled - context will be pruned to reduce noise[/green]")
        console.print(f"[blue]📊 Rank window: {args.rank_window} candidates, Pruning threshold: {args.pruning_threshold}[/blue]")
        console.print(f"[blue]📊 Similarity threshold: {args.confidence_threshold} (similarity score >= threshold required)[/blue]")
        console.print(f"[blue]📊 Relevance check: LLM will verify context actually answers the question[/blue]")
    else:
        console.print("[yellow]⚠️  Elasticsearch long-term memory disabled - no previous conversations available[/yellow]")
    
    # Initialize Tavily (only if not disabled) - added AFTER Elasticsearch
    if not args.no_tavily:
        tavily = TavilySearch(max_results=3)
        tools.append(tavily)
        console.print("[green]🌐 Tavily web search enabled - agent can access recent information[/green]")
    else:
        console.print("[yellow]🌐 Tavily web search disabled - agent will use only training data and long-term memory[/yellow]")
    
    # Initialize chat memory (Note: This is in-memory only, not persistent)
    memory = MemorySaver()
    
    # Create a LangGraph agent
    agent = create_react_agent(model=llm, tools=tools, checkpointer=memory)
    
    return agent, memory, tools

