"""Elasticsearch indexing functions for Elasticsearch Agent."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from elasticsearch import Elasticsearch
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console


def extract_messages_from_checkpoints(checkpoints, thread_id: str) -> List[Dict[str, Any]]:
    """
    Extract all messages from checkpoints and prepare them for indexing.
    
    Parameters:
        checkpoints: List of checkpoint tuples
        thread_id: Thread ID for the conversation
        
    Returns:
        List of dictionaries containing message data ready for indexing
    """
    messages_to_index = []
    seen_message_ids = set()  # Avoid indexing duplicate messages
    
    for checkpoint_tuple in checkpoints:
        checkpoint = checkpoint_tuple.checkpoint
        checkpoint_id = checkpoint.get("id", "")
        timestamp = checkpoint.get("ts", datetime.now().isoformat())
        messages = checkpoint["channel_values"].get("messages", [])
        
        for message in messages:
            # Skip if we've already indexed this message
            if message.id in seen_message_ids:
                continue
            seen_message_ids.add(message.id)
            
            # Determine message type
            if isinstance(message, HumanMessage):
                message_type = "human"
            elif isinstance(message, AIMessage):
                message_type = "ai"
            else:
                message_type = "other"
            
            # Create text for embedding
            text_content = message.content
            
            # Store message data
            message_data = {
                "text": text_content,
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "timestamp": timestamp,
                "message_type": message_type,
                "message_id": message.id,
                "content": text_content
            }
            
            messages_to_index.append(message_data)
    
    return messages_to_index


def summarize_conversation(checkpoints, thread_id: str, llm, verbose: bool = False, console: Optional[Console] = None) -> str:
    """
    Summarize the entire conversation from checkpoints into a single coherent document.
    
    Parameters:
        checkpoints: List of checkpoint tuples
        thread_id: Thread ID for the conversation
        llm: LLM instance for summarization
        verbose: Whether to show verbose output
        console: Rich console for output
        
    Returns:
        A summarized string of the entire conversation
    """
    try:
        # Extract all messages in chronological order
        all_messages = []
        seen_message_ids = set()
        
        for checkpoint_tuple in checkpoints:
            checkpoint = checkpoint_tuple.checkpoint
            timestamp = checkpoint.get("ts", datetime.now().isoformat())
            messages = checkpoint["channel_values"].get("messages", [])
            
            for message in messages:
                if message.id in seen_message_ids:
                    continue
                seen_message_ids.add(message.id)
                
                if isinstance(message, HumanMessage):
                    all_messages.append(f"User ({timestamp}): {message.content}")
                elif isinstance(message, AIMessage):
                    all_messages.append(f"Agent ({timestamp}): {message.content}")
        
        if not all_messages:
            return ""
        
        # Combine all messages into a conversation transcript
        conversation_transcript = "\n\n".join(all_messages)
        
        # Summarize using LLM
        summary_prompt = f"""You are an expert at summarizing conversations. Create a comprehensive summary of the following conversation that preserves all important information, facts, and context.

The summary should:
1. Preserve all key facts, names, preferences, and information discussed
2. Maintain the flow and context of the conversation
3. Be concise but complete
4. Include important details that might be needed for future reference

Conversation:
{conversation_transcript}

Provide a comprehensive summary:"""

        summary = llm.invoke(summary_prompt).content
        
        if verbose and console:
            original_length = len(conversation_transcript)
            summary_length = len(summary)
            reduction_pct = ((original_length - summary_length) / original_length * 100) if original_length > 0 else 0
            console.print(f"[cyan]📝 Conversation summary: {summary_length}/{original_length} chars ({reduction_pct:.1f}% reduction)[/cyan]")
        
        return summary
        
    except Exception as e:
        if console:
            console.print(f"[red]❌ Error summarizing conversation: {e}[/red]")
        import traceback
        if verbose and console:
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return ""


def index_checkpoints_to_elasticsearch(
    checkpoints,
    thread_id: str,
    es_client: Optional[Elasticsearch],
    es_index_name: Optional[str],
    embeddings,
    llm,
    summarize: bool = False,
    verbose: bool = False,
    console: Optional[Console] = None
):
    """
    Index checkpoint messages to Elasticsearch with embeddings.
    Can either index individual messages or a summarized version of the entire conversation.
    
    Parameters:
        checkpoints: List of checkpoint tuples
        thread_id: Thread ID for the conversation
        es_client: Elasticsearch client instance
        es_index_name: Name of the Elasticsearch index
        embeddings: Embeddings model instance
        llm: LLM instance for summarization
        summarize: If True, summarize the entire conversation into a single document. 
                   If False, index each message individually.
        verbose: Whether to show verbose output
        console: Rich console for output
    """
    if not es_client or not es_index_name:
        if console:
            console.print("[red]❌ Elasticsearch not available. Cannot index checkpoints.[/red]")
        return
    
    try:
        # Verify index exists before indexing
        if not es_client.indices.exists(index=es_index_name):
            if console:
                console.print(f"[red]❌ Index {es_index_name} does not exist. Cannot index checkpoints.[/red]")
                console.print("[yellow]⚠️  This should not happen. Please check Elasticsearch connection.[/yellow]")
            return
        
        # Convert generator to list if needed (memory.list() returns a generator)
        if not isinstance(checkpoints, list):
            checkpoints = list(checkpoints)
        
        if not checkpoints:
            if console:
                console.print("[yellow]⚠️  No checkpoints to index.[/yellow]")
            return
        
        if summarize:
            # Summarize the entire conversation into a single document
            if console:
                console.print("\n[yellow]📝 Summarizing conversation...[/yellow]")
            conversation_summary = summarize_conversation(checkpoints, thread_id, llm, verbose, console)
            
            if not conversation_summary:
                if console:
                    console.print("[red]❌ Failed to summarize conversation. Cannot index.[/red]")
                return
            
            # Get the first and last timestamps for metadata
            first_timestamp = checkpoints[0].checkpoint.get("ts", datetime.now().isoformat()) if checkpoints else datetime.now().isoformat()
            last_timestamp = checkpoints[-1].checkpoint.get("ts", datetime.now().isoformat()) if checkpoints else datetime.now().isoformat()
            
            # Create a single document for the summarized conversation
            doc_id = f"{thread_id}_summary_{int(datetime.now().timestamp())}"
            
            # Check if this summary already exists (optional - you might want to allow multiple summaries)
            if es_client.exists(index=es_index_name, id=doc_id):
                if console:
                    console.print("[yellow]⚠️  Summary document already exists. Skipping.[/yellow]")
                return
            
            # Generate embedding for the summary
            if console:
                console.print("[yellow]🔄 Generating embedding and indexing summary to Elasticsearch...[/yellow]")
            embedding = embeddings.embed_query(conversation_summary)
            
            document = {
                "text": conversation_summary,
                "content": conversation_summary,
                "thread_id": thread_id,
                "checkpoint_id": "summary",
                "timestamp": last_timestamp,
                "message_type": "summary",
                "message_id": doc_id,
                "vector": embedding,
                "summary_start": first_timestamp,
                "summary_end": last_timestamp,
                "is_summary": True
            }
            
            try:
                es_client.index(
                    index=es_index_name,
                    id=doc_id,
                    document=document
                )
                if console:
                    console.print(f"\n[green]✅ Summary indexed successfully![/green]")
                    console.print(f"[blue]📊 Summary length: {len(conversation_summary)} characters[/blue]")
                    console.print(f"[blue]📊 Document ID: {doc_id}[/blue]")
                    console.print(f"[blue]📊 Index: {es_index_name}[/blue]")
            except Exception as e:
                if console:
                    console.print(f"[red]❌ Error indexing summary: {e}[/red]")
                import traceback
                if verbose and console:
                    console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        
        else:
            # Index individual messages (original behavior)
            if console:
                console.print("\n[yellow]📦 Extracting messages from checkpoints...[/yellow]")
            messages_to_index = extract_messages_from_checkpoints(checkpoints, thread_id)
            
            if not messages_to_index:
                if console:
                    console.print("[yellow]⚠️  No messages to index.[/yellow]")
                return
            
            if console:
                console.print(f"[blue]📝 Found {len(messages_to_index)} messages to index[/blue]")
            
            # Generate embeddings and index documents
            if console:
                console.print("[yellow]🔄 Generating embeddings and indexing to Elasticsearch...[/yellow]")
            
            indexed_count = 0
            skipped_count = 0
            error_count = 0
            
            for message_data in messages_to_index:
                try:
                    # Generate embedding for the text
                    embedding = embeddings.embed_query(message_data["text"])
                    
                    # Add vector to document
                    document = {
                        **message_data,
                        "vector": embedding,
                        "is_summary": False
                    }
                    
                    # Index document (using message_id as document ID to avoid duplicates)
                    doc_id = f"{thread_id}_{message_data['message_id']}"
                    
                    # Check if document already exists
                    if es_client.exists(index=es_index_name, id=doc_id):
                        skipped_count += 1
                        continue
                    
                    es_client.index(
                        index=es_index_name,
                        id=doc_id,
                        document=document
                    )
                    indexed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if console:
                        console.print(f"[red]❌ Error indexing message {message_data.get('message_id', 'unknown')}: {e}[/red]")
                    continue
            
            # Summary
            if console:
                console.print(f"\n[green]✅ Indexing complete![/green]")
                console.print(f"[blue]📊 Indexed: {indexed_count} | Skipped (duplicates): {skipped_count} | Errors: {error_count}[/blue]")
                console.print(f"[blue]📊 Index: {es_index_name}[/blue]")
        
    except Exception as e:
        if console:
            console.print(f"[red]❌ Error indexing checkpoints to Elasticsearch: {e}[/red]")
        import traceback
        if verbose and console:
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

