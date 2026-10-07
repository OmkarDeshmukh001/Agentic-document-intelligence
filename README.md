# Agentic Document Intelligence

An agentic AI platform for document-grounded reasoning, Retrieval-Augmented Generation (RAG), web search, financial data retrieval, and intelligent tool orchestration.

## Overview

**Agentic Document Intelligence** is an AI-powered document assistant built with **LangGraph and LangChain**. It combines document retrieval with multiple external tools, allowing the system to decide when to retrieve information from an uploaded PDF, search the web, fetch stock information, or perform calculations.

The application maintains **independent chat threads**, allowing each conversation to have its own document context and conversation history.

## Key Features

- 📄 **PDF Question Answering** using Retrieval-Augmented Generation (RAG)
- 🔎 **Semantic document retrieval** using FAISS vector search
- 🧠 **LLM-powered tool selection** using LangGraph
- 🌐 **Web search** for current information
- 📈 **Stock price retrieval** using Alpha Vantage
- 🧮 **Calculator tool** for arithmetic operations
- 💬 **Persistent multi-chat conversations**
- 🗂️ **Independent document context per chat**
- ✏️ **Chat rename and delete**
- ⚡ **Streaming AI responses**
- 🔐 **Environment-based API key configuration**
- 🖥️ **Interactive Streamlit interface**

## Architecture

```text
                    ┌──────────────────────────┐
                    │      Streamlit UI        │
                    │  Chat / PDF Upload / UI  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │        LangGraph         │
                    │   Agentic Orchestration  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             ┌─────────────┐          ┌─────────────┐
             │     LLM     │          │  Tool Node  │
             │ GPT-OSS 120B│          └──────┬──────┘
             └─────────────┘                 │
                                      ┌──────┼──────┬──────┐
                                      ▼      ▼      ▼      ▼
                                     RAG   Search  Stock Calculator
                                      │
                                      ▼
                                ┌────────────┐
                                │    FAISS   │
                                │ Vector DB  │
                                └─────┬──────┘
                                      │
                                      ▼
                            Hugging Face Embeddings
                                      │
                                      ▼
                                     PDF
```

## Tech Stack

### AI / Generative AI

- Python
- Groq
- GPT-OSS 120B
- LangChain
- LangGraph
- LLM Tool Calling

### RAG / Document Processing

- PyPDF
- Recursive Character Text Splitter
- Hugging Face Sentence Transformers
- `all-MiniLM-L6-v2`
- FAISS
- Retrieval-Augmented Generation (RAG)

### Tools & APIs

- DuckDuckGo Search
- Alpha Vantage API
- Custom Calculator Tool

### Application

- Streamlit
- SQLite
- LangGraph SQLite Checkpointing
- Requests
- python-dotenv

## Agentic Workflow

The application uses LangGraph to orchestrate the interaction between the LLM and available tools.

For each user request, the system can determine whether it needs to:

1. Retrieve information from the current chat's uploaded PDF.
2. Search the web for current information.
3. Retrieve stock market information.
4. Perform mathematical calculations.
5. Generate a final response using the retrieved information.

For PDF-related queries, the system automatically associates the retrieval operation with the **current chat thread**, preventing the document context from being mixed across conversations.

## RAG Pipeline

```text
PDF Upload
    │
    ▼
PDF Text Extraction
    │
    ▼
Document Chunking
    │
    ▼
Hugging Face Embeddings
    │
    ▼
FAISS Vector Store
    │
    ▼
Similarity Retrieval
    │
    ▼
Relevant Context
    │
    ▼
LLM
    │
    ▼
Grounded Answer
```

## Project Structure

```text
Agentic-document-intelligence/
│
├── langgraph_backend.py
├── streamlit_frontend.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/OmkarDeshmukh001/Agentic-document-intelligence.git
cd Agentic-document-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Git Bash:**

```bash
source venv/Scripts/activate
```

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
ALPHAVANTAGE_API_KEY=your_alpha_vantage_api_key
```

Never commit the `.env` file to GitHub.

## Run the Application

Start the Streamlit application:

```bash
streamlit run streamlit_frontend.py
```

The application will open in your browser.

## Example Use Cases

### Document Intelligence

Upload a PDF and ask:

```text
Summarize the document.
What are the key concepts?
Explain the methodology used in the document.
What are the main conclusions?
```

### Web Research

```text
What are the latest developments in generative AI?
```

The system can use web search when current information is required.

### Financial Information

```text
What is the latest price of AAPL?
```

The system can use the Alpha Vantage tool to retrieve stock information.

### Mathematical Reasoning

```text
Calculate 125 * 48.
```

The calculator tool can handle supported arithmetic operations.

## Engineering Highlights

- **Agentic tool orchestration** with LangGraph
- **Stateful conversations** using LangGraph checkpointing
- **Thread-aware RAG retrieval**
- **Semantic vector search** with FAISS
- **Local embedding model** using Sentence Transformers
- **Streaming LLM responses**
- **Modular tool architecture**
- **Persistent conversation state** using SQLite

## Future Improvements

- Persistent FAISS indexes across application restarts
- Support for multiple documents per conversation
- Document management and metadata persistence
- MCP-based tool integration
- Authentication and user accounts
- Additional document formats
- Improved retrieval and reranking
- Deployment using Docker and cloud infrastructure

## Author

**Omkar Deshmukh**

AI & Machine Learning Engineering Student

GitHub: https://github.com/OmkarDeshmukh001
