"""Main entry point for Elasticsearch Agent."""

import os
import warnings
from typing import Optional, Any

# Disable tokenizers parallelism warning (set before any tokenizers are loaded)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_core.messages import HumanMessage, AIMessage
from elasticsearch import Elasticsearch
from rich.console import Console

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import modules
from .config.settings import get_args
from .config.constants import get_embedding_dimension
from .models.llm import initialize_llm
from .models.embeddings import initialize_embeddings
from .models.provence import initialize_provence
from .storage.elasticsearch_client import initialize_elasticsearch
from .storage.elasticsearch_retrieval import check_elasticsearch_with_confidence
from .storage.elasticsearch_indexing import index_checkpoints_to_elasticsearch
from .agents.agent_factory import create_agent
from .utils.display import setup_console, process_chunks
from .utils.checkpoints import process_checkpoints


def main():
    """Main function that orchestrates the agent."""
    # Parse arguments
    args = get_args()
    
    # Setup console
    console = setup_console()
    
    # Initialize models
    llm = initialize_llm()
    embeddings, embedding_model, embedding_dimension = initialize_embeddings()
    provence_model = initialize_provence(args, console)
    
    # Initialize Elasticsearch
    es_client, es_index_name, es_is_new_index = initialize_elasticsearch(embedding_dimension, console)
    
    if es_client and es_index_name:
        console.print("[green]✅ Connected to Elasticsearch[/green]")
        if es_is_new_index:
            console.print("[blue]ℹ️  This is a new index - no previous conversations stored yet[/blue]")
        else:
            console.print("[blue]ℹ️  Previous conversations may be available in long-term memory[/blue]")
    else:
        console.print("[yellow]⚠️  Continuing without Elasticsearch persistence and retrieval...[/yellow]")
    
    # Create agent
    agent, memory, tools = create_agent(
        llm, es_client, es_index_name, embeddings, provence_model, args, console
    )
    
    # Main loop
    while True:
        # Get the user's question and display it in the terminal
        user_question = input("\nUser:\n")

        # Check if the user wants to quit the chat
        if user_question.lower() == "quit":
            console.print("\nAgent:\nHave a nice day! :wave:\n", style="black on white")
            
            # Ask if user wants to store checkpoints to Elasticsearch
            if es_client:
                store_choice = input("\n💾 Do you want to store this conversation to Elasticsearch? (yes/no): ").strip().lower()
                if store_choice in ["yes", "y"]:
                    # Ask if user wants to summarize or store individual messages
                    summarize_choice = input("\n📝 How would you like to store the conversation?\n  1. Summarize into a single document (recommended for better retrieval)\n  2. Store individual messages\n  Enter choice (1/2, default=1): ").strip()
                    
                    summarize = True  # Default to summarize
                    if summarize_choice == "2":
                        summarize = False
                        console.print("[blue]📦 Will store individual messages...[/blue]")
                    else:
                        console.print("[blue]📝 Will summarize conversation into a single document...[/blue]")
                    
                    # Get all checkpoints for this thread
                    checkpoints = memory.list({"configurable": {"thread_id": "1"}})
                    index_checkpoints_to_elasticsearch(
                        checkpoints, "1", es_client, es_index_name, embeddings, llm,
                        summarize=summarize, verbose=args.verbose, console=console
                    )
                else:
                    console.print("[yellow]⚠️  Conversation not stored. Checkpoints will be lost when session ends.[/yellow]")
            else:
                console.print("[yellow]⚠️  Elasticsearch not available. Checkpoints will be lost when session ends.[/yellow]")
            
            break

        # STEP 1: Check Elasticsearch first with confidence threshold
        use_elasticsearch = False
        elasticsearch_context = ""
        max_score = 0.0
        
        if es_client and es_index_name:
            if args.verbose:
                console.print(f"\n[bold yellow]🔍 Checking Elasticsearch first...[/bold yellow]")
            
            has_results, context_or_msg, score = check_elasticsearch_with_confidence(
                user_question, es_client, es_index_name, embeddings, llm, provence_model, args, console
            )
            
            if has_results:
                use_elasticsearch = True
                elasticsearch_context = context_or_msg
                max_score = score
                
                if args.verbose:
                    console.print(f"[green]✅ Using Elasticsearch results (similarity: {max_score:.4f}, relevance check passed)[/green]")
            else:
                if args.verbose:
                    console.print(f"[yellow]⚠️  Elasticsearch not sufficient (score: {score:.4f} < {args.confidence_threshold}). Will use Tavily.[/yellow]")
        else:
            if args.verbose:
                console.print(f"[yellow]⚠️  Elasticsearch not available. Will use Tavily.[/yellow]")
        
        # STEP 2: Generate answer based on source
        if use_elasticsearch:
            # Use Elasticsearch context directly with LLM
            if args.verbose:
                console.print(f"[yellow]🤖 Generating answer from Elasticsearch context...[/yellow]")
            
            answer_prompt = f"""Based on the following context from previous conversations, please answer the user's question.

Context:
{elasticsearch_context}

Question: {user_question}

Answer:"""
            
            try:
                answer = llm.invoke(answer_prompt).content
                console.print(f"\nAgent:\n{answer}", style="black on white")
                
                # Store the interaction in memory for checkpoint tracking
                # Add to agent's memory for consistency
                agent.invoke(
                    {"messages": [HumanMessage(content=user_question), AIMessage(content=answer)]},
                    {"configurable": {"thread_id": "1"}},
                )
            except Exception as e:
                console.print(f"[red]❌ Error generating answer: {e}[/red]")
                console.print("[yellow]⚠️  Falling back to agent with Tavily...[/yellow]")
                use_elasticsearch = False
        
        # STEP 3: If Elasticsearch didn't work, use agent with Tavily
        if not use_elasticsearch:
            if args.verbose:
                console.print(f"[yellow]🌐 Using agent with Tavily for web search...[/yellow]")
            
            # Use the stream method of the LangGraph agent to get the agent's answer
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=user_question)]},
                {"configurable": {"thread_id": "1"}},
            ):
                # Process the chunks from the agent
                process_chunks(chunk, console)

        # Only process and display checkpoints if verbose mode is enabled
        if args.verbose:
            # List all checkpoints that match a given configuration
            checkpoints = memory.list({"configurable": {"thread_id": "1"}})
            # Process the checkpoints
            process_checkpoints(checkpoints, console)


if __name__ == "__main__":
    main()

