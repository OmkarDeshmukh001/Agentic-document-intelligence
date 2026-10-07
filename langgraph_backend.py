from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import (
    InjectedState,
    ToolNode,
    tools_condition,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# ============================================================
# HUGGING FACE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# IN-MEMORY PDF RETRIEVERS
# ============================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(
    thread_id: Optional[str],
):
    """
    Get the PDF retriever associated with a chat thread.
    """

    if not thread_id:
        return None

    return _THREAD_RETRIEVERS.get(
        str(thread_id)
    )


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Read a PDF, split it into chunks, create a FAISS
    vector store, and associate the retriever with
    the current chat thread.
    """

    if not file_bytes:
        raise ValueError(
            "No PDF data received."
        )

    thread_id = str(
        thread_id
    )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            temp_file.write(
                file_bytes
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(
            temp_path
        )

        documents = loader.load()

        if not documents:
            raise ValueError(
                "PDF contains no readable content."
            )

        # ----------------------------------------------------
        # Split PDF into chunks
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(
            documents
        )

        if not chunks:
            raise ValueError(
                "Could not create text chunks from PDF."
            )

        # ----------------------------------------------------
        # Create FAISS vector store
        # ----------------------------------------------------

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        # ----------------------------------------------------
        # Create retriever
        # ----------------------------------------------------

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
            },
        )

        # ----------------------------------------------------
        # Associate retriever with current chat
        # ----------------------------------------------------

        _THREAD_RETRIEVERS[
            thread_id
        ] = retriever

        # ----------------------------------------------------
        # Store metadata
        # ----------------------------------------------------

        metadata = {
            "filename": (
                filename
                or os.path.basename(
                    temp_path
                )
            ),
            "documents": len(
                documents
            ),
            "chunks": len(
                chunks
            ),
        }

        _THREAD_METADATA[
            thread_id
        ] = metadata

        return metadata

    finally:

        # ----------------------------------------------------
        # Delete temporary PDF
        # ----------------------------------------------------

        if temp_path:

            try:
                os.remove(
                    temp_path
                )
            except OSError:
                pass


# ============================================================
# WEB SEARCH TOOL
# ============================================================

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ============================================================
# CALCULATOR TOOL
# ============================================================

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform basic arithmetic.

    Supported operations:
    add
    sub
    mul
    div
    """

    try:

        operation = (
            operation
            .lower()
            .strip()
        )

        if operation == "add":

            result = (
                first_num
                + second_num
            )

        elif operation == "sub":

            result = (
                first_num
                - second_num
            )

        elif operation == "mul":

            result = (
                first_num
                * second_num
            )

        elif operation == "div":

            if second_num == 0:

                return {
                    "error": (
                        "Division by zero "
                        "is not allowed."
                    )
                }

            result = (
                first_num
                / second_num
            )

        else:

            return {
                "error": (
                    f"Unsupported operation "
                    f"'{operation}'. "
                    "Use add, sub, mul, or div."
                )
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# STOCK PRICE TOOL
# ============================================================

@tool
def get_stock_price(
    symbol: str,
) -> dict:
    """
    Fetch the latest stock quote using Alpha Vantage.
    """

    api_key = os.getenv(
        "ALPHAVANTAGE_API_KEY"
    )

    if not api_key:

        return {
            "error": (
                "ALPHAVANTAGE_API_KEY is not "
                "configured in the .env file."
            )
        }

    symbol = (
        symbol
        .upper()
        .strip()
    )

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )

    try:

        response = requests.get(
            url,
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# RAG TOOL
# ============================================================

@tool
def rag_tool(
    query: str,
    thread_id: Annotated[
        str,
        InjectedState("thread_id"),
    ],
) -> dict:
    """
    Retrieve relevant information from the PDF
    belonging to the current chat.

    thread_id is injected automatically by LangGraph.
    The LLM does not need to provide it.
    """

    if not thread_id:

        return {
            "error": (
                "No chat thread was identified."
            ),
            "query": query,
        }

    thread_id = str(
        thread_id
    )

    # --------------------------------------------------------
    # Get retriever belonging to this chat
    # --------------------------------------------------------

    retriever = _get_retriever(
        thread_id
    )

    if retriever is None:

        return {
            "error": (
                "No document is available "
                "in this chat. "
                "Upload a PDF first."
            ),
            "query": query,
        }

    try:

        # ----------------------------------------------------
        # Retrieve relevant chunks
        # ----------------------------------------------------

        results = retriever.invoke(
            query
        )

        if not results:

            return {
                "query": query,
                "context": [],
                "metadata": [],
                "source_file": (
                    _THREAD_METADATA
                    .get(
                        thread_id,
                        {},
                    )
                    .get(
                        "filename"
                    )
                ),
            }

        # ----------------------------------------------------
        # Extract context
        # ----------------------------------------------------

        context = [
            doc.page_content
            for doc in results
        ]

        # ----------------------------------------------------
        # Extract metadata
        # ----------------------------------------------------

        metadata = [
            doc.metadata
            for doc in results
        ]

        # ----------------------------------------------------
        # Return RAG result
        # ----------------------------------------------------

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": (
                _THREAD_METADATA
                .get(
                    thread_id,
                    {},
                )
                .get(
                    "filename"
                )
            ),
        }

    except Exception as e:

        return {
            "error": str(e),
            "query": query,
        }


# ============================================================
# TOOLS
# ============================================================

tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]


# ============================================================
# LLM WITH TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    thread_id: str


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(
    state: ChatState,
    config=None,
):
    """
    Main LangGraph chatbot node.
    """

    # --------------------------------------------------------
    # Get thread ID from state
    # --------------------------------------------------------

    thread_id = state.get(
        "thread_id"
    )

    # --------------------------------------------------------
    # Fallback to config
    # --------------------------------------------------------

    if not thread_id:

        if (
            config
            and isinstance(
                config,
                dict,
            )
        ):

            thread_id = (
                config
                .get(
                    "configurable",
                    {},
                )
                .get(
                    "thread_id"
                )
            )

    # --------------------------------------------------------
    # System message
    # --------------------------------------------------------

    system_message = SystemMessage(
        content=f"""
You are a helpful AI assistant.

Current chat thread:
{thread_id}


AVAILABLE TOOLS:

1. rag_tool
   Use this for questions about the uploaded PDF.
   The current chat's PDF is automatically provided
   to the tool.

2. search
   Use this for current web information.

3. get_stock_price
   Use this for stock-price questions.

4. calculator
   Use this for arithmetic.


PDF RULES:

- If the user's question is about the uploaded PDF,
  use rag_tool.

- The current chat's document is automatically
  provided to rag_tool.

- You do NOT need to provide a thread_id
  to rag_tool.

- Do not pretend you know PDF content if no PDF
  has been uploaded.

- If the user asks something that depends on
  the PDF and no PDF is available, tell the user
  to upload a PDF.

- Use the retrieved PDF context when answering
  document-related questions.

- Do not invent information that is not supported
  by the retrieved document.


GENERAL RULES:

- Use calculator for arithmetic when useful.
- Use web search for current information.
- Use get_stock_price for stock questions.
- Answer clearly and concisely.
"""
    )

    # --------------------------------------------------------
    # Build message list
    # --------------------------------------------------------

    messages = [
        system_message,
        *state["messages"],
    ]

    # --------------------------------------------------------
    # Invoke LLM
    # --------------------------------------------------------

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    # --------------------------------------------------------
    # Return state
    # --------------------------------------------------------

    return {
        "messages": [
            response
        ],
        "thread_id": str(
            thread_id
        ),
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# SQLITE DATABASE
# ============================================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False,
)


# ============================================================
# LANGGRAPH CHECKPOINTER
# ============================================================

checkpointer = SqliteSaver(
    conn=conn
)


# ============================================================
# CHAT METADATA TABLE
# ============================================================

conn.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_metadata (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
    """
)

conn.commit()


# ============================================================
# GET CHAT TITLE
# ============================================================

def get_chat_title(
    thread_id: str,
) -> str:

    cursor = conn.execute(
        """
        SELECT title
        FROM chat_metadata
        WHERE thread_id = ?
        """,
        (
            str(thread_id),
        ),
    )

    row = cursor.fetchone()

    if row:

        return row[0]

    return "New Chat"


# ============================================================
# RENAME CHAT
# ============================================================

def rename_chat(
    thread_id: str,
    title: str,
):
    """
    Create or update a chat title.
    """

    thread_id = str(
        thread_id
    )

    title = title.strip()

    if not title:

        title = "New Chat"

    conn.execute(
        """
        INSERT INTO chat_metadata (
            thread_id,
            title
        )
        VALUES (?, ?)

        ON CONFLICT(thread_id)
        DO UPDATE SET
            title = excluded.title
        """,
        (
            thread_id,
            title,
        ),
    )

    conn.commit()


# ============================================================
# DELETE CHAT
# ============================================================

def delete_chat(
    thread_id: str,
):
    """
    Delete conversation, metadata,
    and in-memory PDF data.
    """

    thread_id = str(
        thread_id
    )

    # --------------------------------------------------------
    # Delete LangGraph checkpoints
    # --------------------------------------------------------

    checkpointer.delete_thread(
        thread_id
    )

    # --------------------------------------------------------
    # Delete metadata
    # --------------------------------------------------------

    conn.execute(
        """
        DELETE FROM chat_metadata
        WHERE thread_id = ?
        """,
        (
            thread_id,
        ),
    )

    conn.commit()

    # --------------------------------------------------------
    # Delete PDF retriever
    # --------------------------------------------------------

    _THREAD_RETRIEVERS.pop(
        thread_id,
        None,
    )

    _THREAD_METADATA.pop(
        thread_id,
        None,
    )


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(
    ChatState
)


# ============================================================
# ADD NODES
# ============================================================

graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_node(
    "tools",
    tool_node,
)


# ============================================================
# GRAPH EDGES
# ============================================================

graph.add_edge(
    START,
    "chat_node",
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge(
    "tools",
    "chat_node",
)


# ============================================================
# COMPILE GRAPH
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# RETRIEVE ALL CHAT THREADS
# ============================================================

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(
        None
    ):

        config = checkpoint.config

        configurable = config.get(
            "configurable",
            {},
        )

        thread_id = configurable.get(
            "thread_id"
        )

        if thread_id:

            all_threads.add(
                str(thread_id)
            )

    return list(
        all_threads
    )


# ============================================================
# CHECK THREAD DOCUMENT
# ============================================================

def thread_has_document(
    thread_id: str,
) -> bool:

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )


# ============================================================
# GET THREAD DOCUMENT METADATA
# ============================================================

def thread_document_metadata(
    thread_id: str,
) -> dict:

    return _THREAD_METADATA.get(
        str(thread_id),
        {},
    )
