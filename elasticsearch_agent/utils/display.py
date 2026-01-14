"""Display utilities for Elasticsearch Agent."""

from rich.console import Console


def setup_console() -> Console:
    """
    Set up Rich console for output formatting.
    
    Returns:
        Rich Console instance
    """
    return Console()


def process_chunks(chunk, console: Console):
    """
    Processes a chunk from the agent and displays information about tool calls or the agent's answer.
    
    Args:
        chunk: Chunk from agent stream
        console: Rich console for output
    """
    # Check if the chunk contains an agent's message
    if "agent" in chunk:
        # Iterate over the messages in the chunk
        for message in chunk["agent"]["messages"]:
            # Check if the message contains tool calls
            if "tool_calls" in message.additional_kwargs:
                # Extract all the tool calls
                tool_calls = message.additional_kwargs["tool_calls"]

                # Iterate over the tool calls
                for tool_call in tool_calls:
                    # Extract the tool name
                    tool_name = tool_call["function"]["name"]

                    # Extract the tool arguments
                    tool_arguments = eval(tool_call["function"]["arguments"])
                    
                    # Handle different tool argument structures
                    if tool_name == "search_long_term_memory":
                        tool_query = tool_arguments.get("query", "")
                        console.print(
                            f"\n🧠 The agent is searching [on deep_sky_blue1]long-term memory (Elasticsearch)[/on deep_sky_blue1] for: [on deep_sky_blue1]{tool_query}[/on deep_sky_blue1]...",
                            style="deep_sky_blue1",
                        )
                    elif "query" in tool_arguments:
                        tool_query = tool_arguments["query"]
                        console.print(
                            f"\nThe agent is calling the tool [on deep_sky_blue1]{tool_name}[/on deep_sky_blue1] with the query [on deep_sky_blue1]{tool_query}[/on deep_sky_blue1]. Please wait for the agent's answer[deep_sky_blue1]...[/deep_sky_blue1]",
                            style="deep_sky_blue1",
                        )
                    else:
                        console.print(
                            f"\nThe agent is calling the tool [on deep_sky_blue1]{tool_name}[/on deep_sky_blue1]...",
                            style="deep_sky_blue1",
                        )
            else:
                # If the message doesn't contain tool calls, extract and display the agent's answer
                agent_answer = message.content
                console.print(f"\nAgent:\n{agent_answer}", style="black on white")

