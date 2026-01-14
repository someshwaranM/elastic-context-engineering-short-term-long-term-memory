# Managing Agentic Memory with Elasticsearch

## 📖 Overview

A comprehensive solution for managing agentic memory using a dual-memory architecture that combines **short-term memory** (checkpointers) with **long-term memory** (Elasticsearch). This system enables AI agents to maintain context across sessions, intelligently retrieve relevant past conversations, and optimize memory through summarization and context pruning.

### Memory Architecture

- 🧠 **Long-Term Memory (Elasticsearch)**: Persistent storage of conversation history across sessions with semantic search capabilities
- 💾 **Short-Term Memory (Checkpointers)**: In-memory session-based storage using LangGraph's MemorySaver for active conversation context
- 📝 **Conversation Summarization**: Each conversation is automatically summarized before storage to reduce duplication and improve retrieval
- ✂️ **Context Pruning**: Intelligent removal of irrelevant content from retrieved context using Provence reranker for optimal relevance
- 🔍 **Semantic Retrieval**: Advanced similarity search with confidence scoring to find the most relevant past conversations
- ✅ **Relevance Validation**: LLM-based verification ensures retrieved context actually answers the user's query
- 🌐 **Live Web Search**: Optional Tavily integration for accessing current information when long-term memory doesn't have recent data

---

## 🧠 How Memory Management Works

### Short-Term Memory (Checkpointers)

During an active conversation session, the agent uses LangGraph's `MemorySaver` to maintain short-term memory. This in-memory storage:
- Tracks the current conversation flow in real-time
- Maintains context within a single session
- Stores checkpoints that can be persisted to long-term memory when the session ends

### Long-Term Memory (Elasticsearch)

When conversations end, they can be stored in Elasticsearch for persistent long-term memory:

1. **Storage Options**:
   - **Summarized**: The entire conversation is summarized into a single document (recommended for better retrieval)
   - **Individual Messages**: Each message is stored separately with embeddings

2. **Retrieval Process**:
   - Semantic search finds relevant past conversations based on query similarity
   - Confidence scoring determines if retrieved context is sufficient
   - Context pruning removes irrelevant sentences using Provence reranker
   - Summarization reduces duplication in retrieved context
   - Relevance checking validates that context actually answers the query

3. **Optimization Techniques**:
   - **Context Pruning**: Provence reranker filters out irrelevant sentences, keeping only content that directly relates to the query
   - **Context Summarization**: LLM-powered summarization reduces token usage and focuses on key information
   - **Confidence Thresholds**: Configurable similarity scores determine when to use long-term memory vs. live web search

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- Elasticsearch instance (cloud or local)
- OpenAI API key
- (Optional) Tavily API key for web search

### Environment Variables

Create a `.env` file in the root directory:

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

1. **Create and activate virtual environment:**
   ```bash
   python -m venv my-venv
   source my-venv/bin/activate  # On Windows: my-venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the agent:**
   ```bash
   python -m elasticsearch_agent.main
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

## 📊 Memory Management Flow

### During Conversation (Short-Term Memory)

1. **Active Session**: User interacts with the agent
2. **Checkpoint Storage**: Each interaction is stored in short-term memory (checkpointers)
3. **Context Maintenance**: Agent maintains conversation context within the session
4. **Real-Time Access**: Current conversation history is immediately available

### Retrieving Past Conversations (Long-Term Memory)

1. **Query Processing**: User asks a question
2. **Semantic Search**: System searches Elasticsearch for semantically similar past conversations
3. **Confidence Scoring**: Similarity scores determine relevance (default threshold: 0.7)
4. **Context Pruning**: Provence reranker removes irrelevant sentences, keeping only query-relevant content
5. **Context Summarization**: LLM summarizes retrieved context to reduce duplication and token usage
6. **Relevance Validation**: LLM verifies that the retrieved context actually answers the query
7. **Answer Generation**: If validation passes, answer is generated using the optimized context
8. **Fallback to Web Search**: If long-term memory doesn't have sufficient information, Tavily provides current data

### Storing Conversations (Long-Term Memory Persistence)

When a conversation session ends:
1. **User Choice**: User decides whether to store the conversation to Elasticsearch
2. **Storage Mode Selection**:
   - **Summarized** (Recommended): Entire conversation is summarized into a single document for better retrieval
   - **Individual Messages**: Each message stored separately with embeddings
3. **Embedding Generation**: Vector embeddings are created for semantic search
4. **Indexing**: Documents are indexed to Elasticsearch for future retrieval across sessions

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'elasticsearch_agent'
```
**Solution**: Ensure you're running from the repository root directory where `elasticsearch_agent/` is located, or ensure `elasticsearch_agent` is in your Python path.

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

## 📋 TODO / Roadmap

### ✅ Completed Features

- [x] **Context Pruning**: Implemented Provence reranker for removing irrelevant content from retrieved context
- [x] **Context Summarization**: LLM-powered summarization to reduce duplication and focus on relevant information
- [x] **Elasticsearch Long-Term Storage**: Persistent memory storage using Elasticsearch for conversation history across sessions
- [x] **Checkpointers for Short-Term Storage**: In-memory checkpoint storage using LangGraph's MemorySaver for session-based memory
- [x] **Tavily Integration**: Live web search integration for accessing current information and real-time data
- [x] **Relevance Checking**: LLM-based validation to ensure retrieved context actually answers the query
- [x] **Confidence Scoring**: Similarity-based scoring system to determine when to use Elasticsearch vs Tavily

### 🚧 Pending Work

- [ ] **Subagent Architecture**: Implement subagent system with specialized agents for different tasks
  - [ ] Memory Agent: Dedicated agent for Elasticsearch long-term memory operations
  - [ ] Research Agent: Specialized agent for Tavily web search operations
  - [ ] Supervisor Router: LLM-based routing system to direct queries to appropriate subagents
  - [ ] Context Isolation: Separate thread contexts for each subagent to prevent context contamination

---

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

---

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Provence Reranker](https://huggingface.co/naver/provence-reranker-debertav3-v1)

---

**Version**: 1.0.0

