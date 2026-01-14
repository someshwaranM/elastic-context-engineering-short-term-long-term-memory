"""Settings and argument parsing for Elasticsearch Agent."""

import argparse
from dotenv import load_dotenv


def get_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='LangGraph agent with Elasticsearch persistent memory and Provence pruning'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed checkpoint and retrieval information'
    )
    parser.add_argument(
        '--no-tavily',
        action='store_true',
        help='Disable Tavily web search tool (agent will use only training data)'
    )
    parser.add_argument(
        '--pruning-threshold',
        type=float,
        default=0.3,
        help='Pruning threshold for Provence reranker (0.1=conservative, 0.5=aggressive, default=0.3)'
    )
    parser.add_argument(
        '--rank-window',
        type=int,
        default=10,
        help='Number of candidates to retrieve from Elasticsearch before pruning (default=10)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for Elasticsearch results. If max score >= threshold, use Elasticsearch. Otherwise use Tavily (default=0.7)'
    )
    
    # Load environment variables
    load_dotenv()
    
    return parser.parse_args()

