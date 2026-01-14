# Elasticsearch Agent - LangGraph Agent with Persistent Memory

## 📖 Overview

**Elasticsearch Agent** is a powerful, modular LangGraph agent that combines the capabilities of LangChain's LangGraph framework with Elasticsearch for persistent long-term memory. The agent features intelligent context retrieval, automatic context pruning, and relevance checking to provide accurate, context-aware responses.

### Key Features

- 🧠 **Persistent Long-Term Memory**: Store and retrieve conversation history using Elasticsearch
- 🔍 **Intelligent Retrieval**: Semantic search with confidence scoring and relevance validation
- ✂️ **Context Pruning**: Automatic removal of irrelevant content using Provence reranker
- 📝 **Context Summarization**: LLM-powered summarization to reduce duplication
- ✅ **Relevance Checking**: Ensures retrieved context actually answers the query
- 🌐 **Web Search Integration**: Optional Tavily integration for current information
- 🎯 **Modular Architecture**: Clean, maintainable codebase organized by functionality

---

## 🏗️ Architecture

### Directory Structure

```
python/
├── elasticsearch_agent/               # Main package
│   ├── __init__.py                    # Package initialization
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                # Argument parsing, env vars
│   │   └── constants.py               # Embedding dimensions, defaults
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm.py                     # ChatOpenAI initialization
│   │   ├── embeddings.py              # OpenAIEmbeddings setup
│   │   └── provence.py                # Provence reranker loading
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── elasticsearch_client.py   # ES connection & index creation
│   │   ├── elasticsearch_retrieval.py # Retrieval functions
│   │   └── elasticsearch_indexing.py # Indexing functions
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── context_pruning.py         # Context pruning
│   │   ├── context_summarization.py   # Context summarization
│   │   └── relevance_check.py         # Relevance validation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent_factory.py           # Agent creation
│   │   └── tools.py                   # Tool definitions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── checkpoints.py             # Checkpoint processing
│   │   └── display.py                 # Display utilities
│   └── main.py                        # Main entry point
│
└── README.md                          # This file
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **config** | Configuration and constants | `get_args()`, `get_embedding_dimension()` |
| **models** | LLM, embeddings, and reranker initialization | `initialize_llm()`, `initialize_embeddings()`, `initialize_provence()` |
| **storage** | Elasticsearch operations | `initialize_elasticsearch()`, `retrieve_from_elasticsearch()`, `index_checkpoints_to_elasticsearch()` |
| **processing** | Context processing and validation | `prune_with_provence()`, `summarize_context()`, `check_context_relevance()` |
| **agents** | Agent and tool creation | `create_agent()`, `create_elasticsearch_memory_tool()` |
| **utils** | Display and checkpoint utilities | `process_chunks()`, `process_checkpoints()`, `setup_console()` |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- Elasticsearch instance (cloud or local)
- OpenAI API key
- (Optional) Tavily API key for web search

### Environment Variables

Create a `.env` file in the `python/` directory:

```bash
# Required
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxx
ELASTICSEARCH_URL=https://your-cluster.es.region.cloud.es.io:9243
ELASTICSEARCH_API_KEY=your-api-key-here

# Optional
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Default: text-embedding-3-small
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxxx  # For web search
```

### Installation Steps

1. **Navigate to the python directory:**
   ```bash
   cd python
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv my-venv
   source my-venv/bin/activate  # On Windows: my-venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the agent:**
   ```bash
   python -m elasticsearch_agent.main
   ```

   Or from the `elasticsearch_agent` directory:
   ```bash
   python elasticsearch_agent/main.py
   ```

---

## ⚡ Quick Start

Once installed, here's how to get started in 3 steps:

### Step 1: Start the Agent
```bash
python -m elasticsearch_agent.main
```

You'll see:
```
✅ Connected to Elasticsearch
🧠 Elasticsearch long-term memory enabled
🌐 Tavily web search enabled

User:
```

### Step 2: Have Your First Conversation
```
User:
Hi, my name is Sarah and I'm interested in machine learning.

Agent:
Hello Sarah! Nice to meet you. Machine learning is a fascinating field! 
How can I help you today?

User:
What's the weather in New York today?

Agent:
[Agent uses Tavily to get current weather information]

User:
quit
```

### Step 3: Store the Conversation
When you type `quit`, you'll be asked:
```
💾 Do you want to store this conversation to Elasticsearch? (yes/no): yes
📝 How would you like to store the conversation?
  1. Summarize into a single document (recommended for better retrieval)
  2. Store individual messages
  Enter choice (1/2, default=1): 1
✅ Summary indexed successfully!
```

### Step 4: Test Memory in Next Session
Start the agent again and ask about previous conversations:
```
User:
What's my name?

Agent:
Your name is Sarah! We discussed earlier that you're interested in machine learning.
```

**That's it!** The agent now remembers your conversations and can retrieve them in future sessions.

> 💡 **Tip**: Use `--verbose` flag to see detailed processing information:
> ```bash
> python -m elasticsearch_agent.main --verbose
> ```

---

## 📚 Function Reference

### Core Functions

#### 1. `initialize_elasticsearch()`

**Location**: `elasticsearch_agent/storage/elasticsearch_client.py`

**Purpose**: Initialize Elasticsearch client and create index if needed.

**Parameters**:
- `embedding_dimension` (int): Dimension of embeddings (for index naming)
- `console` (Console): Rich console for output

**Returns**: `Tuple[Optional[Elasticsearch], Optional[str], bool]`

**Example**:
```python
from elasticsearch_agent.storage.elasticsearch_client import initialize_elasticsearch
from elasticsearch_agent.utils.display import setup_console

console = setup_console()
es_client, index_name, is_new = initialize_elasticsearch(1536, console)
```

---

#### 2. `prune_with_provence()`

**Location**: `elasticsearch_agent/processing/context_pruning.py`

**Purpose**: Prune context using Provence reranker model to remove irrelevant sentences.

**Parameters**:
- `query` (str): User's query/question
- `context` (str): Original context to prune
- `provence_model` (Optional[Any]): Provence reranker model instance
- `threshold` (float): Relevance threshold (0.1=conservative, 0.5=aggressive)
- `verbose` (bool): Whether to show verbose output
- `console` (Optional[Console]): Rich console for output

**Returns**: `str` - Pruned context with only relevant sentences

---

#### 3. `summarize_context()`

**Location**: `elasticsearch_agent/processing/context_summarization.py`

**Purpose**: Summarize context using LLM to reduce duplication and focus on relevant information.

**Parameters**:
- `query` (str): User's query/question
- `context` (str): Context to summarize
- `llm`: LLM instance for summarization
- `verbose` (bool): Whether to show verbose output
- `console` (Optional[Console]): Rich console for output

**Returns**: `str` - Summarized context

---

#### 4. `retrieve_from_elasticsearch()`

**Location**: `elasticsearch_agent/storage/elasticsearch_retrieval.py`

**Purpose**: Retrieve context from Elasticsearch with score-based ranking.

**Parameters**:
- `query` (str): Search query
- `es_client` (Optional[Elasticsearch]): Elasticsearch client instance
- `es_index_name` (Optional[str]): Name of the Elasticsearch index
- `embeddings`: Embeddings model instance
- `rank_window` (int): Number of candidates to retrieve before ranking
- `k` (int): Number of results to return (default: 5)
- `verbose` (bool): Whether to show verbose output
- `console` (Optional[Console]): Rich console for output

**Returns**: `Tuple[List[Dict[str, Any]], str]`

---

#### 5. `check_context_relevance()`

**Location**: `elasticsearch_agent/processing/relevance_check.py`

**Purpose**: Use LLM to check if retrieved context actually answers the user's query.

**Parameters**:
- `query` (str): User's query
- `context` (str): Retrieved context from Elasticsearch
- `llm`: LLM instance for relevance checking
- `verbose` (bool): Whether to show verbose output
- `console` (Optional[Console]): Rich console for output

**Returns**: `Tuple[bool, float]` - (is_relevant, relevance_score)

---

#### 6. `check_elasticsearch_with_confidence()`

**Location**: `elasticsearch_agent/storage/elasticsearch_retrieval.py`

**Purpose**: Check Elasticsearch for relevant context and return confidence score. Includes LLM-based relevance checking.

**Parameters**:
- `query` (str): User's query
- `es_client` (Optional[Elasticsearch]): Elasticsearch client instance
- `es_index_name` (Optional[str]): Name of the Elasticsearch index
- `embeddings`: Embeddings model instance
- `llm`: LLM instance for relevance checking
- `provence_model` (Optional[Any]): Provence reranker model
- `args`: Parsed command line arguments
- `console` (Optional[Console]): Rich console for output

**Returns**: `Tuple[bool, str, float]` - (has_results, context_or_message, max_score)

---

#### 7. `index_checkpoints_to_elasticsearch()`

**Location**: `elasticsearch_agent/storage/elasticsearch_indexing.py`

**Purpose**: Index checkpoint messages to Elasticsearch with embeddings. Can either index individual messages or a summarized version.

**Parameters**:
- `checkpoints`: List of checkpoint tuples
- `thread_id` (str): Thread ID for the conversation
- `es_client` (Optional[Elasticsearch]): Elasticsearch client instance
- `es_index_name` (Optional[str]): Name of the Elasticsearch index
- `embeddings`: Embeddings model instance
- `llm`: LLM instance for summarization
- `summarize` (bool): If True, summarize into a single document; if False, index individual messages
- `verbose` (bool): Whether to show verbose output
- `console` (Optional[Console]): Rich console for output

---

## 🎮 Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--verbose` | `-v` | Show detailed checkpoint and retrieval information | `False` |
| `--no-tavily` | - | Disable Tavily web search tool | `False` |
| `--pruning-threshold` | - | Pruning threshold for Provence reranker (0.1=conservative, 0.5=aggressive) | `0.3` |
| `--rank-window` | - | Number of candidates to retrieve from Elasticsearch before pruning | `10` |
| `--confidence-threshold` | - | Confidence threshold for Elasticsearch results. If max score >= threshold, use Elasticsearch. Otherwise use Tavily | `0.7` |

### Usage Examples

**Basic usage:**
```bash
python -m elasticsearch_agent.main
```

**Verbose mode:**
```bash
python -m elasticsearch_agent.main --verbose
```

**Custom thresholds:**
```bash
python -m elasticsearch_agent.main \
  --verbose \
  --pruning-threshold 0.3 \
  --rank-window 10 \
  --confidence-threshold 0.7
```

**Without Tavily (long-term memory only):**
```bash
python -m elasticsearch_agent.main --no-tavily
```

---

## 💬 Example Conversations

### Example 1: First Conversation (No Memory Yet)

When you first run the agent, it won't have any stored memories, so it will use Tavily for web search:

```
User:
Hi, my name is Alice and I'm a software engineer.

Agent:
Hello Alice! Nice to meet you. It's great to know you're a software engineer. 
How can I assist you today with your work or anything else you'd like to discuss?

User:
What's the weather in San Francisco today?

Agent:
Today in San Francisco, the weather is expected to be partly cloudy with:
- Maximum Temperature: Approximately 18°C
- Minimum Temperature: Around 12°C
- Condition: Partly Cloudy
- Wind: 15 kph from the west

User:
quit

Agent:
Have a nice day! 👋

💾 Do you want to store this conversation to Elasticsearch? (yes/no): yes
📝 How would you like to store the conversation?
  1. Summarize into a single document (recommended for better retrieval)
  2. Store individual messages
  Enter choice (1/2, default=1): 1
✅ Summary indexed successfully!
```

### Example 2: Using Stored Memory

After storing conversations, the agent can retrieve and use previous context:

```
User:
What's my name?

Agent:
Your name is Alice! We discussed earlier that you're a software engineer.

User:
What did we talk about earlier?

Agent:
Earlier in our conversation, we discussed:
- You introduced yourself as Alice, a software engineer
- We checked the weather in San Francisco, which was partly cloudy with temperatures around 12-18°C

Is there anything else you'd like to know?
```

### Example 3: Current Information (Tavily Fallback)

For questions about current events or real-time data, the agent uses Tavily:

```
User:
What are the latest developments in AI this week?

Agent:
[The agent uses Tavily to search for recent AI news and provides current information]

User:
What's the current stock price of Apple?

Agent:
[The agent uses Tavily to get real-time stock information]
```

### Example 4: Memory + Current Information

The agent intelligently combines stored memory with current information:

```
User:
What's the weather in San Francisco today? And remind me what we discussed yesterday.

Agent:
Today in San Francisco, the weather is [current weather from Tavily].

Regarding our previous conversation, we discussed [retrieved from Elasticsearch memory]:
- You introduced yourself as Alice, a software engineer
- We checked the weather in San Francisco previously
```

### Example 5: Verbose Mode

Run with `--verbose` to see detailed processing:

```bash
python -m elasticsearch_agent.main --verbose
```

```
User:
What's my favorite programming language?

🔍 Checking Elasticsearch first...
📝 Pruning context with Provence reranker...
📝 Summarizing context to reduce duplication...
✅ Using Elasticsearch results (similarity: 0.8523, relevance check passed)
🤖 Generating answer from Elasticsearch context...

Agent:
Based on our previous conversation, you mentioned that Python is your favorite programming language!
```

### Quick Start Guide

1. **First Run**: Start a conversation and introduce yourself
   ```
   User: Hi, I'm Bob and I love machine learning.
   ```

2. **Store the Conversation**: When done, type `quit` and choose to store the conversation
   ```
   User: quit
   💾 Do you want to store this conversation to Elasticsearch? (yes/no): yes
   ```

3. **Second Run**: Start a new session and ask about previous conversations
   ```
   User: What's my name?
   Agent: Your name is Bob!
   ```

4. **Try Current Information**: Ask about recent events or real-time data
   ```
   User: What's the latest news about Python 3.13?
   ```

5. **Combine Both**: The agent automatically uses memory when available, and Tavily for current info
   ```
   User: What did we talk about? And what's the weather today?
   ```

---

## 📊 How It Works

### Query Processing Flow

1. **User Query**: User submits a question
2. **Elasticsearch Check**: System checks if Elasticsearch has relevant past conversations
3. **Confidence Scoring**: Similarity scores are calculated for retrieved documents
4. **Threshold Check**: If score >= confidence threshold, proceed with Elasticsearch results
5. **Context Pruning**: Irrelevant sentences are removed using Provence reranker
6. **Context Summarization**: Context is summarized to reduce duplication
7. **Relevance Check**: LLM validates that context actually answers the query
8. **Answer Generation**: If all checks pass, answer is generated from Elasticsearch context
9. **Fallback**: If Elasticsearch doesn't have sufficient results, agent uses Tavily web search
10. **Storage**: Conversation is stored in checkpoints for future retrieval

### Indexing Flow

When a conversation ends:
1. User chooses to store conversation to Elasticsearch
2. Option to summarize or store individual messages
3. Embeddings are generated for each message or summary
4. Documents are indexed to Elasticsearch for future retrieval

---

## 🧪 Testing

The modular structure makes testing easier:

```python
# Example: Testing context pruning
from elasticsearch_agent.processing.context_pruning import prune_with_provence
from unittest.mock import Mock

# Mock dependencies
mock_model = Mock()
mock_console = Mock()

result = prune_with_provence(
    query="test",
    context="test context",
    provence_model=mock_model,
    threshold=0.3,
    verbose=False,
    console=mock_console
)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'elasticsearch_agent'
```
**Solution**: Run from the `python/` directory or ensure `elasticsearch_agent` is in your Python path.

**2. Elasticsearch Connection Errors**
```
ConnectionError: Failed to connect to Elasticsearch
```
**Solution**: Check your `ELASTICSEARCH_URL` and `ELASTICSEARCH_API_KEY` in `.env`.

**3. Provence Model Loading Fails**
```
⚠️  Provence reranker not available
```
**Solution**: This is non-critical. The system will continue without pruning. Ensure `transformers` and `nltk` are installed.

---

## 📝 Module Documentation

### config Module

**Purpose**: Configuration management and constants.

**Files**:
- `settings.py`: Argument parsing and environment variable loading
- `constants.py`: Embedding dimensions and default values

### models Module

**Purpose**: Initialize LLM, embeddings, and reranker models.

**Files**:
- `llm.py`: ChatOpenAI initialization
- `embeddings.py`: OpenAIEmbeddings setup
- `provence.py`: Provence reranker loading

### storage Module

**Purpose**: Elasticsearch operations (connection, retrieval, indexing).

**Files**:
- `elasticsearch_client.py`: ES connection and index creation
- `elasticsearch_retrieval.py`: Retrieval and confidence checking
- `elasticsearch_indexing.py`: Indexing checkpoints and messages

### processing Module

**Purpose**: Context processing and validation.

**Files**:
- `context_pruning.py`: Provence-based context pruning
- `context_summarization.py`: LLM-based context summarization
- `relevance_check.py`: LLM-based relevance validation

### agents Module

**Purpose**: Agent and tool creation.

**Files**:
- `agent_factory.py`: LangGraph agent creation
- `tools.py`: Elasticsearch memory tool definition

### utils Module

**Purpose**: Display and checkpoint utilities.

**Files**:
- `display.py`: Chunk processing and console setup
- `checkpoints.py`: Checkpoint display utilities

---

## 🎯 Design Principles

1. **Modularity**: Code organized into logical, focused modules
2. **Dependency Injection**: Pass dependencies as parameters for testability
3. **Separation of Concerns**: Each module has single responsibility
4. **Type Safety**: Type hints throughout for better IDE support
5. **Documentation**: Comprehensive docstrings and README

---

## 📄 License

See LICENSE file for details.

---

## 🤝 Contributing

When contributing:

1. Follow the existing module structure
2. Maintain dependency injection patterns
3. Add type hints to all functions
4. Update this README if adding new functions
5. Write tests for new functionality

---

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Provence Reranker](https://huggingface.co/naver/provence-reranker-debertav3-v1)

---

**Version**: 1.0.0

