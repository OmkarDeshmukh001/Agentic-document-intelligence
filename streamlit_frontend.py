import uuid

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph_backend import (
    chatbot,
    delete_chat,
    get_chat_title,
    ingest_pdf,
    rename_chat,
    retrieve_all_threads,
    thread_document_metadata,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================

def generate_thread_id():
    return str(
        uuid.uuid4()
    )


def generate_chat_title(
    message,
):
    title = " ".join(
        message.strip().split()
    )

    if len(title) > 35:
        title = (
            title[:35]
            .rstrip()
            + "..."
        )

    return title or "New Chat"


def add_thread(
    thread_id,
):
    thread_id = str(
        thread_id
    )

    if (
        thread_id
        not in st.session_state[
            "chat_threads"
        ]
    ):

        st.session_state[
            "chat_threads"
        ].append(
            thread_id
        )


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state[
        "thread_id"
    ] = thread_id

    st.session_state[
        "message_history"
    ] = []

    st.session_state[
        "ingested_docs"
    ][
        thread_id
    ] = {}

    add_thread(
        thread_id
    )


def load_conversation(
    thread_id,
):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": str(
                    thread_id
                )
            }
        }
    )

    return state.values.get(
        "messages",
        [],
    )


def pretty_tool(
    name,
):

    return (
        name
        .replace(
            "_",
            " ",
        )
        .capitalize()
    )


# ============================================================
# SESSION STATE
# ============================================================

if (
    "message_history"
    not in st.session_state
):

    st.session_state[
        "message_history"
    ] = []


if (
    "thread_id"
    not in st.session_state
):

    st.session_state[
        "thread_id"
    ] = generate_thread_id()


if (
    "chat_threads"
    not in st.session_state
):

    st.session_state[
        "chat_threads"
    ] = retrieve_all_threads()


if (
    "ingested_docs"
    not in st.session_state
):

    st.session_state[
        "ingested_docs"
    ] = {}


# ============================================================
# CURRENT CHAT
# ============================================================

current_thread = str(
    st.session_state[
        "thread_id"
    ]
)

add_thread(
    current_thread
)

current_document = (
    thread_document_metadata(
        current_thread
    )
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📚 AI Document Assistant"
)


# ============================================================
# NEW CHAT
# ============================================================

if st.sidebar.button(
    "＋ New Chat",
    use_container_width=True,
):

    reset_chat()

    st.rerun()


# ============================================================
# DOCUMENT
# ============================================================

st.sidebar.subheader(
    "Document"
)


if current_document:

    filename = (
        current_document.get(
            "filename",
            "Document",
        )
    )

    st.sidebar.write(
        f"📄 {filename}"
    )

else:

    st.sidebar.caption(
        "No PDF uploaded."
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=[
        "pdf"
    ],
    key=f"uploader-{current_thread}",
)


if uploaded_pdf:

    existing_docs = (
        st.session_state[
            "ingested_docs"
        ].setdefault(
            current_thread,
            {},
        )
    )

    if (
        uploaded_pdf.name
        in existing_docs
    ):

        st.sidebar.info(
            "PDF already uploaded."
        )

    else:

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=current_thread,
                    filename=uploaded_pdf.name,
                )

                existing_docs[
                    uploaded_pdf.name
                ] = summary

                st.rerun()

            except Exception as e:

                st.sidebar.error(
                    f"PDF processing failed: {e}"
                )


# ============================================================
# CHAT HISTORY
# ============================================================

st.sidebar.subheader(
    "Chats"
)


threads = [
    str(thread_id)
    for thread_id in st.session_state[
        "chat_threads"
    ]
][::-1]


for thread_id in threads:

    title = get_chat_title(
        thread_id
    )

    if title == "New Chat":
        continue

    col1, col2 = (
        st.sidebar.columns(
            [5, 1]
        )
    )

    # --------------------------------------------------------
    # OPEN CHAT
    # --------------------------------------------------------

    with col1:

        if st.button(
            title,
            key=f"chat-{thread_id}",
            use_container_width=True,
        ):

            st.session_state[
                "thread_id"
            ] = thread_id

            history = []

            for message in load_conversation(
                thread_id
            ):

                if isinstance(
                    message,
                    HumanMessage,
                ):

                    role = "user"

                elif isinstance(
                    message,
                    AIMessage,
                ):

                    role = "assistant"

                else:

                    continue

                if (
                    isinstance(
                        message.content,
                        str,
                    )
                    and message.content.strip()
                ):

                    history.append(
                        {
                            "role": role,
                            "content": message.content,
                        }
                    )

            st.session_state[
                "message_history"
            ] = history

            st.session_state[
                "ingested_docs"
            ].setdefault(
                thread_id,
                {},
            )

            st.rerun()

    # --------------------------------------------------------
    # CHAT MENU
    # --------------------------------------------------------

    with col2:

        with st.popover(
            "⋮"
        ):

            new_title = st.text_input(
                "Chat name",
                value=title,
                key=f"title-{thread_id}",
            )

            # ------------------------------------------------
            # Rename
            # ------------------------------------------------

            if st.button(
                "Rename",
                key=f"rename-{thread_id}",
            ):

                if new_title.strip():

                    rename_chat(
                        thread_id,
                        new_title,
                    )

                    st.rerun()

            # ------------------------------------------------
            # Delete
            # ------------------------------------------------

            if st.button(
                "Delete",
                key=f"delete-{thread_id}",
            ):

                delete_chat(
                    thread_id
                )

                st.session_state[
                    "chat_threads"
                ] = [
                    t
                    for t
                    in st.session_state[
                        "chat_threads"
                    ]
                    if str(t)
                    != thread_id
                ]

                st.session_state[
                    "ingested_docs"
                ].pop(
                    thread_id,
                    None,
                )

                if (
                    str(
                        st.session_state[
                            "thread_id"
                        ]
                    )
                    == thread_id
                ):

                    reset_chat()

                st.rerun()


# ============================================================
# MAIN AREA
# ============================================================

st.title(
    "AI Document Assistant"
)


if current_document:

    st.caption(
        f"📄 {current_document.get('filename')}"
    )

else:

    st.caption(
        "Upload a PDF and ask questions about it."
    )


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state[
    "message_history"
]:

    st.write("")

    st.subheader(
        "How can I help?"
    )

    if current_document:

        suggestions = [
            "Summarize this document",
            "List the key points",
            "What are the main conclusions?",
        ]

    else:

        suggestions = [
            "What can you help me with?",
            "Explain RAG in simple terms",
            "What tools do you have?",
        ]

    for text in suggestions:

        if st.button(
            text,
            key=f"suggest-{text}",
        ):

            st.session_state[
                "pending_prompt"
            ] = text

            st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state[
    "message_history"
]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# INPUT
# ============================================================

typed_input = st.chat_input(
    "Ask a question..."
)


user_input = (
    st.session_state.pop(
        "pending_prompt",
        None,
    )
    or typed_input
)


# ============================================================
# CHAT
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Create title for first message
    # --------------------------------------------------------

    if (
        get_chat_title(
            current_thread
        )
        == "New Chat"
    ):

        rename_chat(
            current_thread,
            generate_chat_title(
                user_input
            ),
        )

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.write(
            user_input
        )

    # --------------------------------------------------------
    # LangGraph configuration
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": current_thread,
        },
        "metadata": {
            "thread_id": current_thread,
        },
        "run_name": "chat_turn",
    }

    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        tool_status = {
            "box": None
        }

        def stream_response():

            for (
                message_chunk,
                metadata,
            ) in chatbot.stream(

                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ],

                    # IMPORTANT:
                    # Pass current chat ID into
                    # LangGraph state.
                    "thread_id": current_thread,
                },

                config=config,

                stream_mode="messages",
            ):

                # --------------------------------------------
                # Tool execution
                # --------------------------------------------

                if isinstance(
                    message_chunk,
                    ToolMessage,
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool",
                    )

                    if (
                        tool_status[
                            "box"
                        ]
                        is None
                    ):

                        tool_status[
                            "box"
                        ] = st.status(
                            f"Using {pretty_tool(tool_name)}..."
                        )

                # --------------------------------------------
                # AI response
                # --------------------------------------------

                if isinstance(
                    message_chunk,
                    AIMessage,
                ):

                    if isinstance(
                        message_chunk.content,
                        str,
                    ):

                        yield (
                            message_chunk.content
                        )

        assistant_message = (
            st.write_stream(
                stream_response()
            )
        )

        # ----------------------------------------------------
        # Complete tool status
        # ----------------------------------------------------

        if tool_status[
            "box"
        ]:

            tool_status[
                "box"
            ].update(
                label="Completed",
                state="complete",
            )

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )

    # --------------------------------------------------------
    # Refresh sidebar after first message
    # --------------------------------------------------------

    if len(
        st.session_state[
            "message_history"
        ]
    ) == 2:

        st.rerun()
