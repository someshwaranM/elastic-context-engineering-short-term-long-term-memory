"""Checkpoint processing utilities for Elasticsearch Agent."""

from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console


def process_checkpoints(checkpoints, console: Console):
    """
    Processes a list of checkpoints and displays relevant information.
    
    Args:
        checkpoints: List of checkpoint tuples
        console: Rich console for output
    """
    console.print("\n==========================================================\n")

    for idx, checkpoint_tuple in enumerate(checkpoints):
        # Extract key information about the checkpoint
        checkpoint = checkpoint_tuple.checkpoint
        messages = checkpoint["channel_values"].get("messages", [])

        # Display checkpoint information
        console.print(f"[white]Checkpoint:[/white]")
        console.print(f"[black]Timestamp: {checkpoint['ts']}[/black]")
        console.print(f"[black]Checkpoint ID: {checkpoint['id']}[/black]")

        # Display checkpoint messages
        for message in messages:
            if isinstance(message, HumanMessage):
                console.print(
                    f"[bright_magenta]User: {message.content}[/bright_magenta] [bright_cyan](Message ID: {message.id})[/bright_cyan]"
                )
            elif isinstance(message, AIMessage):
                console.print(
                    f"[bright_magenta]Agent: {message.content}[/bright_magenta] [bright_cyan](Message ID: {message.id})[/bright_cyan]"
                )

        console.print("")

    console.print("==========================================================")

