"""Streamlit app for the local LLM Wiki."""

from __future__ import annotations

import uuid

import streamlit as st
import streamlit.components.v1 as components

from auth import AuthUser, LocalAuthManager
from rag_pipeline import DocumentRecord, RAGPipeline, RetrievedChunk
from utils import (
    build_workspace_config,
    format_bytes,
    load_config,
    setup_logging,
    truncate_text,
)


LOGGER = setup_logging()
APP_TITLE = "Local LLM Wiki"
SAMPLE_QUESTIONS = [
    "What are the internal support hours?",
    "How often are usage metrics refreshed?",
    "Who owns the platform roadmap?",
    "Explain retrieval augmented generation in simple terms.",
    "What security guidelines should employees follow?",
]


def inject_styles() -> None:
    """Apply a more polished visual treatment to the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 197, 66, 0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(34, 119, 102, 0.14), transparent 28%),
                linear-gradient(180deg, #f7f1e8 0%, #f2ede3 55%, #ece6db 100%);
            color: #1d2b28;
            font-family: 'IBM Plex Sans', 'Avenir Next', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Fraunces', Georgia, serif;
            letter-spacing: -0.03em;
            color: #183530;
        }

        /* --- Sidebar --- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f0e6 0%, #ede7dc 100%);
            border-right: 1px solid rgba(24, 53, 48, 0.10);
        }
        section[data-testid="stSidebar"] * {
            color: #1d2b28 !important;
        }
        section[data-testid="stSidebar"] .stMetric label,
        section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
            color: #183530 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stTextInput label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stFileUploader label {
            color: #384844 !important;
            font-weight: 500;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #183530 !important;
        }

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 252, 247, 0.7);
            border: 1px solid rgba(24, 53, 48, 0.12);
            border-radius: 10px 10px 0 0;
            padding: 0.5rem 1.2rem;
            color: #384844 !important;
            font-weight: 500;
            font-size: 0.92rem;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff;
            color: #183530 !important;
            font-weight: 600;
            border-bottom: 2px solid #2c7a69;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.85);
            color: #183530 !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }

        /* --- Buttons --- */
        .stButton > button {
            background: #2c7a69 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 999px;
            font-weight: 500;
            padding: 0.45rem 1rem;
            transition: background 0.2s;
        }
        .stButton > button:hover {
            background: #1a5c4e !important;
            color: #ffffff !important;
        }
        .stButton > button:active {
            background: #14493d !important;
            color: #ffffff !important;
        }

        /* --- Quick prompt buttons (same style) --- */
        .quick-prompt button {
            border-radius: 999px;
        }

        /* --- Info / Success / Warning / Error alerts --- */
        .stAlert {
            border-radius: 12px;
        }
        [data-testid="stAlert"] {
            color: #1d2b28 !important;
        }

        /* --- Cards --- */
        .hero-card,
        .panel-card,
        .login-card {
            background: rgba(255, 252, 247, 0.92);
            border: 1px solid rgba(24, 53, 48, 0.12);
            border-radius: 22px;
            box-shadow: 0 18px 45px rgba(64, 52, 37, 0.08);
            padding: 1.1rem 1.2rem;
            backdrop-filter: blur(10px);
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1.55fr 1fr 1fr 1fr 1fr;
            gap: 0.9rem;
            margin: 0.5rem 0 1rem 0;
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            color: #6b665f;
            margin-bottom: 0.4rem;
        }
        .hero-title {
            font-family: 'Fraunces', Georgia, serif;
            font-size: 2.35rem;
            line-height: 1;
            margin-bottom: 0.55rem;
            color: #183530;
        }
        .hero-copy {
            color: #384844;
            font-size: 0.98rem;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #103631;
            margin: 0.2rem 0;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #726754;
        }
        .source-pill {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            margin: 0 0.35rem 0.35rem 0;
            border-radius: 999px;
            background: #dce9e3;
            color: #18433b;
            font-size: 0.78rem;
            border: 1px solid rgba(24, 67, 59, 0.12);
        }

        /* --- Chat messages --- */
        .stChatMessage {
            background: rgba(255, 252, 247, 0.88);
            border: 1px solid rgba(24, 53, 48, 0.08);
            border-radius: 18px;
        }

        /* --- Chat input --- */
        .stChatInput textarea {
            color: #1d2b28 !important;
            background: #ffffff !important;
        }

        /* --- Expanders --- */
        .streamlit-expanderHeader {
            color: #183530 !important;
            font-weight: 500;
        }

        /* --- Subheaders --- */
        [data-testid="stSubheader"] {
            color: #183530 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_auth_manager() -> LocalAuthManager:
    """Initialize the local auth manager."""
    return LocalAuthManager(load_config())


@st.cache_resource(show_spinner=False)
def get_pipeline(workspace_name: str) -> RAGPipeline:
    """Initialize a workspace-scoped RAG pipeline once per Streamlit session."""
    config = build_workspace_config(load_config(), workspace_name)
    return RAGPipeline(config)


def initialize_session_state() -> None:
    """Set up required Streamlit session keys."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_context", [])
    st.session_state.setdefault("last_answer", "")
    st.session_state.setdefault("last_question", "")
    st.session_state.setdefault("selected_sources", [])
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("chat_model", None)
    st.session_state.setdefault("top_k", None)
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("workspace_name", None)
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("conversation_title", "New Conversation")
    st.session_state.setdefault("last_export_path", "")
    st.session_state.setdefault("last_saved_note", "")


def set_authenticated_user(user: AuthUser) -> None:
    """Store a signed-in user in session state."""
    st.session_state.auth_user = {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "workspaces": user.workspaces,
    }


def current_user() -> AuthUser:
    """Return the currently signed-in user from session state."""
    payload = st.session_state.auth_user or {}
    return AuthUser(
        username=str(payload.get("username", "local-admin")),
        display_name=str(payload.get("display_name", "Local Admin")),
        role=str(payload.get("role", "owner")),
        workspaces=[str(item) for item in payload.get("workspaces", ["General"])],
    )


def create_new_conversation() -> None:
    """Reset the UI state for a fresh conversation."""
    st.session_state.conversation_id = uuid.uuid4().hex
    st.session_state.conversation_title = "New Conversation"
    st.session_state.messages = []
    st.session_state.last_context = []
    st.session_state.last_answer = ""
    st.session_state.last_question = ""


def save_current_conversation(pipeline: RAGPipeline) -> None:
    """Persist the current conversation to disk."""
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = uuid.uuid4().hex
    pipeline.save_conversation(
        st.session_state.conversation_id,
        st.session_state.conversation_title,
        st.session_state.messages,
    )


def load_conversation_into_session(pipeline: RAGPipeline, conversation_id: str) -> None:
    """Load a persisted conversation into session state."""
    payload = pipeline.load_conversation(conversation_id)
    st.session_state.conversation_id = str(payload.get("conversation_id", conversation_id))
    st.session_state.conversation_title = str(payload.get("title", "Conversation"))
    st.session_state.messages = list(payload.get("messages", []))
    st.session_state.last_context = []
    st.session_state.last_answer = ""
    st.session_state.last_question = ""


def render_login(auth_manager: LocalAuthManager) -> bool:
    """Render the login screen when authentication is enabled."""
    if not auth_manager.enabled:
        set_authenticated_user(auth_manager.get_default_user())
        return True

    if st.session_state.auth_user:
        return True

    st.markdown(
        """
        <div class="login-card">
            <div class="hero-kicker">Protected Workspace</div>
            <div class="hero-title">Sign In</div>
            <div class="hero-copy">Use a configured local account to access team workspaces, saved conversations, and compiled wiki artifacts.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("login-form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            user = auth_manager.authenticate(username, password)
            if user is None:
                st.error("Invalid username or password.")
            else:
                set_authenticated_user(user)
                st.rerun()
    return False


def ensure_workspace_selected(auth_manager: LocalAuthManager) -> str:
    """Ensure there is an active workspace for the current user."""
    user = current_user()
    workspace_options = auth_manager.get_workspace_options(user)
    if st.session_state.workspace_name not in workspace_options:
        st.session_state.workspace_name = workspace_options[0]
    return str(st.session_state.workspace_name)


def render_sidebar(
    pipeline: RAGPipeline,
    config: dict[str, object],
    auth_manager: LocalAuthManager,
) -> None:
    """Render upload, workspace, conversation, and runtime controls."""
    user = current_user()
    workspace_options = auth_manager.get_workspace_options(user)

    st.sidebar.header("Workspace")
    st.sidebar.caption(f"Signed in as {user.display_name} • {user.role}")

    selected_workspace = st.sidebar.selectbox(
        "Team space",
        options=workspace_options,
        index=workspace_options.index(st.session_state.workspace_name),
    )
    if selected_workspace != st.session_state.workspace_name:
        st.session_state.workspace_name = selected_workspace
        st.session_state.selected_sources = []
        create_new_conversation()
        st.rerun()

    if auth_manager.enabled and st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.auth_user = None
        st.session_state.workspace_name = None
        create_new_conversation()
        st.rerun()

    conversations = pipeline.list_conversations()
    conversation_ids = ["__new__"] + [item.conversation_id for item in conversations]
    conversation_labels = {"__new__": "New Conversation"}
    for item in conversations:
        conversation_labels[item.conversation_id] = (
            f"{item.title} • {item.message_count} messages"
        )

    current_selector = (
        st.session_state.conversation_id
        if st.session_state.conversation_id in conversation_ids
        else "__new__"
    )
    selected_conversation = st.sidebar.selectbox(
        "Conversation",
        options=conversation_ids,
        index=conversation_ids.index(current_selector),
        format_func=lambda item: conversation_labels[item],
    )
    if selected_conversation == "__new__" and current_selector != "__new__":
        create_new_conversation()
        st.rerun()
    if (
        selected_conversation != "__new__"
        and selected_conversation != st.session_state.conversation_id
    ):
        load_conversation_into_session(pipeline, selected_conversation)
        st.rerun()

    if st.sidebar.button("Start New Conversation", use_container_width=True):
        create_new_conversation()
        st.rerun()

    if (
        st.session_state.conversation_id
        and st.sidebar.button("Delete Conversation", use_container_width=True)
    ):
        pipeline.delete_conversation(st.session_state.conversation_id)
        create_new_conversation()
        st.rerun()

    st.session_state.conversation_title = st.sidebar.text_input(
        "Conversation title",
        value=st.session_state.conversation_title,
    )

    stats = pipeline.get_stats()
    st.sidebar.metric("Documents", stats["documents"])
    st.sidebar.metric("Learned notes", stats["learned_notes"])
    st.sidebar.metric("Chunks", stats["chunks"])
    st.sidebar.metric("Storage", format_bytes(stats["storage_bytes"]))

    try:
        model_options = pipeline.list_available_models()
    except Exception:
        model_options = []

    default_model = str(config["ollama"]["chat_model"])
    available_models = model_options or [default_model]
    if st.session_state.chat_model not in available_models:
        st.session_state.chat_model = (
            default_model if default_model in available_models else available_models[0]
        )

    st.session_state.chat_model = st.sidebar.selectbox(
        "Chat model",
        options=available_models,
        index=available_models.index(st.session_state.chat_model),
        help="Choose the Ollama model used to answer questions.",
    )
    default_top_k = int(config["retrieval"]["top_k"])
    st.session_state.top_k = st.sidebar.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=int(st.session_state.top_k or default_top_k),
    )

    library = pipeline.get_library()
    source_records = [record for record in library if record.kind == "source"]
    all_sources = [record.source for record in source_records]
    st.session_state.selected_sources = st.sidebar.multiselect(
        "Limit search to sources",
        options=all_sources,
        default=[
            source for source in st.session_state.selected_sources if source in all_sources
        ],
        help="Leave empty to search across the full workspace.",
    )

    uploaded_files = st.sidebar.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, and Markdown.",
    )
    if st.sidebar.button("Ingest Uploaded Documents", use_container_width=True):
        if not uploaded_files:
            st.sidebar.warning("Upload at least one file before ingesting.")
        else:
            with st.sidebar:
                with st.spinner("Embedding and indexing documents..."):
                    for uploaded_file in uploaded_files:
                        saved_path = pipeline.save_uploaded_file(
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                        )
                        chunk_count = pipeline.ingest_file(saved_path)
                        st.success(
                            f"Indexed {uploaded_file.name} into {chunk_count} chunks."
                        )
            st.rerun()

    if st.sidebar.button("Load Sample Documents", use_container_width=True):
        with st.sidebar:
            with st.spinner("Loading bundled sample documents..."):
                results = pipeline.ingest_sample_documents()
                for source_name, chunk_count in results:
                    st.success(f"Loaded {source_name} into {chunk_count} chunks.")
        st.rerun()

    if st.sidebar.button("Export Transcript", use_container_width=True):
        if not st.session_state.messages:
            st.sidebar.warning("There is no conversation to export yet.")
        else:
            exported_path = pipeline.export_chat_transcript(
                st.session_state.messages,
                st.session_state.conversation_title,
                user.display_name,
            )
            st.session_state.last_export_path = str(exported_path)
            st.sidebar.success(f"Saved transcript to {exported_path.name}")

    if st.sidebar.button("Save Last Answer to Wiki", use_container_width=True):
        if not st.session_state.last_answer or not st.session_state.last_context:
            st.sidebar.warning("Ask a question first so there is something to learn from.")
        else:
            note_path = pipeline.save_answer_to_wiki(
                st.session_state.last_question,
                st.session_state.last_answer,
                st.session_state.last_context,
                user.display_name,
            )
            st.session_state.last_saved_note = str(note_path)
            st.sidebar.success(f"Saved learned note {note_path.name}")
            st.rerun()

    if st.sidebar.button("Reset Knowledge Base", use_container_width=True):
        pipeline.reset()
        create_new_conversation()
        st.sidebar.success("Knowledge base cleared.")
        st.rerun()

    st.sidebar.subheader("Indexed Sources")
    if source_records:
        for record in source_records[:8]:
            st.sidebar.caption(f"{record.source} • {record.chunk_count} chunks")
    else:
        st.sidebar.info("No source documents indexed yet.")


def render_quick_prompts() -> None:
    """Render suggested prompts for first-time users."""
    st.markdown("### Try a question")
    columns = st.columns(len(SAMPLE_QUESTIONS[:3]))
    for column, prompt in zip(columns, SAMPLE_QUESTIONS[:3]):
        with column:
            if st.button(prompt, use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()


def render_chat_history() -> None:
    """Render previously submitted chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_context_panel(chunks: list[RetrievedChunk]) -> None:
    """Show retrieved chunks that informed the latest answer."""
    st.subheader("Retrieved Context")
    if not chunks:
        st.info("Context snippets will appear here after your first question.")
        return

    for index, chunk in enumerate(chunks, start=1):
        label = (
            f"{index}. {chunk.source} | chunk {chunk.chunk_index} "
            f"| relevance {chunk.relevance * 100:.0f}%"
        )
        with st.expander(label, expanded=index == 1):
            st.caption(f"Citation: [{chunk.citation}]")
            st.progress(max(0.0, min(1.0, chunk.relevance)))
            st.write(chunk.content)


def render_library(records: list[DocumentRecord], pipeline: RAGPipeline) -> None:
    """Display indexed documents and per-document actions."""
    st.subheader("Library")
    if not records:
        st.info("Upload documents or load the sample set to populate your wiki.")
        return

    for record in records:
        with st.container(border=True):
            col1, col2, col3 = st.columns([2.2, 1.2, 0.8])
            col1.markdown(f"**{record.source}**")
            col1.caption(
                f"{record.kind.replace('_', ' ').title()} • {record.file_type.upper()} • {record.chunk_count} chunks • {format_bytes(record.size_bytes)}"
            )
            col2.caption(f"Indexed {record.ingested_at.replace('T', ' ')[:19]} UTC")
            if col3.button(
                "Delete", key=f"delete-{record.source}", use_container_width=True
            ):
                pipeline.delete_source(record.source)
                if record.source in st.session_state.selected_sources:
                    st.session_state.selected_sources.remove(record.source)
                st.rerun()


def handle_question(pipeline: RAGPipeline) -> None:
    """Accept a chat prompt, retrieve context, and stream the answer."""
    question = st.session_state.pop("pending_question", None) or st.chat_input(
        "Ask a question about your uploaded documents"
    )
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    if st.session_state.conversation_title == "New Conversation":
        st.session_state.conversation_title = truncate_text(question, 48)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            retrieved_chunks = pipeline.retrieve_chunks(
                question,
                top_k=st.session_state.top_k,
                source_filters=st.session_state.selected_sources or None,
            )
        except Exception as exc:
            LOGGER.exception("Retrieval failed")
            st.error(str(exc))
            return

        try:
            answer = st.write_stream(
                pipeline.stream_answer(
                    question,
                    retrieved_chunks,
                    chat_model=st.session_state.chat_model,
                )
            )
        except Exception:
            LOGGER.exception("Streaming answer failed, falling back to non-streaming mode")
            answer = pipeline.generate_answer(
                question,
                retrieved_chunks,
                chat_model=st.session_state.chat_model,
            )
            st.markdown(answer)

        st.caption("Sources used")
        sources = []
        for chunk in retrieved_chunks:
            if chunk.source not in sources:
                sources.append(chunk.source)
        st.markdown(
            " ".join(
                f"<span class='source-pill'>{source}</span>" for source in sources
            ),
            unsafe_allow_html=True,
        )

    st.session_state.messages.append({"role": "assistant", "content": str(answer)})
    st.session_state.last_context = retrieved_chunks
    st.session_state.last_answer = str(answer)
    st.session_state.last_question = question
    save_current_conversation(pipeline)


def render_header(pipeline: RAGPipeline) -> None:
    """Display app title, intro, and health status."""
    stats = pipeline.get_stats()
    st.markdown(
        f"""
        <div class="hero-grid">
            <div class="hero-card">
                <div class="hero-kicker">Compiled Local Knowledge</div>
                <div class="hero-title">{APP_TITLE}</div>
                <div class="hero-copy">Upload sources, ask questions, save useful answers back into the wiki, and keep every workspace private on your own machine with Ollama.</div>
            </div>
            <div class="panel-card">
                <div class="metric-label">Source Documents</div>
                <div class="metric-value">{stats['documents']}</div>
                <div class="hero-copy">Raw workspace material</div>
            </div>
            <div class="panel-card">
                <div class="metric-label">Learned Notes</div>
                <div class="metric-value">{stats['learned_notes']}</div>
                <div class="hero-copy">Saved answers that compound</div>
            </div>
            <div class="panel-card">
                <div class="metric-label">Vector Chunks</div>
                <div class="metric-value">{stats['chunks']}</div>
                <div class="hero-copy">Searchable retrieval units</div>
            </div>
            <div class="panel-card">
                <div class="metric-label">Stored Files</div>
                <div class="metric-value">{format_bytes(stats['storage_bytes'])}</div>
                <div class="hero-copy">Workspace footprint</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Active workspace: {pipeline.workspace_name} • conversations, index, log, and learned notes persist under this space."
    )

    try:
        pipeline.check_ollama()
        st.success("Ollama connection healthy")
    except Exception as exc:
        st.warning(
            "Ollama is not reachable right now. Start Ollama before asking questions. "
            f"Details: {exc}"
        )


def render_last_answer_summary(chunks: list[RetrievedChunk]) -> None:
    """Display a compact answer provenance summary."""
    st.subheader("Answer Lens")
    if not chunks:
        st.info("Ask a question to see which documents shaped the answer.")
        return

    unique_sources: list[str] = []
    for chunk in chunks:
        if chunk.source not in unique_sources:
            unique_sources.append(chunk.source)

    st.markdown(
        " ".join(f"<span class='source-pill'>{source}</span>" for source in unique_sources),
        unsafe_allow_html=True,
    )
    st.caption("Top context preview")
    st.write(truncate_text(chunks[0].content, max_chars=280))


def render_wiki_view(pipeline: RAGPipeline) -> None:
    """Render compiled wiki artifacts and learning mechanics."""
    st.subheader("Compiled Wiki")
    st.caption(
        "The workspace maintains generated source pages, an index, a chronological log, and learned notes saved from useful answers."
    )
    index_tab, log_tab, learned_tab, prompts_tab = st.tabs(
        ["Index", "Log", "Learned Notes", "Sample Questions"]
    )
    with index_tab:
        st.markdown(pipeline.get_wiki_index())
    with log_tab:
        st.markdown(pipeline.get_wiki_log())
    with learned_tab:
        pages = pipeline.list_query_pages()
        if not pages:
            st.info("No learned notes yet. Use 'Save Last Answer to Wiki' after a useful answer.")
        else:
            for page in pages[:8]:
                with st.expander(page.name, expanded=False):
                    st.markdown(page.read_text(encoding="utf-8"))
    with prompts_tab:
        st.markdown("### Production-ready sample questions")
        for prompt in SAMPLE_QUESTIONS:
            st.write(f"- {prompt}")
        st.markdown("### Automated learning components")
        st.write("- Uploaded documents create persistent source summary pages under the workspace wiki.")
        st.write("- The app maintains index.md and log.md automatically for each workspace.")
        st.write("- Useful answers can be saved back into the wiki as learned notes and re-ingested for future retrieval.")
        st.write("- Conversations persist per workspace so ongoing work is not lost between sessions.")
        st.write("- The knowledge graph extracts entities and relationships and generates entity wiki pages.")


def render_knowledge_graph_view(pipeline: RAGPipeline) -> None:
    """Render the interactive knowledge graph tab."""
    st.subheader("Knowledge Graph")
    st.caption(
        "Extract entities and relationships from your documents. "
        "Build the graph to see how concepts connect across sources."
    )

    col_build, col_synth = st.columns(2)
    with col_build:
        if st.button("Build Knowledge Graph", use_container_width=True):
            with st.spinner("Extracting entities and relationships from all sources..."):
                try:
                    stats = pipeline.build_knowledge_graph(
                        chat_model=st.session_state.chat_model
                    )
                    st.success(
                        f"Graph built: {stats['entity_count']} entities, "
                        f"{stats['relationship_count']} relationships"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with col_synth:
        if st.button("Synthesize Wiki Pages", use_container_width=True):
            with st.spinner("Running LLM synthesis pass on all source documents..."):
                try:
                    count = pipeline.synthesize_wiki_pages(
                        chat_model=st.session_state.chat_model
                    )
                    st.success(f"Synthesized {count} source pages")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    graph_tab, entities_tab, stats_tab = st.tabs(
        ["Interactive Graph", "Entity Pages", "Graph Stats"]
    )

    with graph_tab:
        graph_html = pipeline.get_knowledge_graph_html()
        if graph_html:
            components.html(graph_html, height=560, scrolling=False)
        else:
            st.info(
                "No knowledge graph built yet. Upload documents and click "
                "'Build Knowledge Graph' to extract entities and relationships."
            )

    with entities_tab:
        entity_pages = pipeline.list_entity_pages()
        if not entity_pages:
            st.info("Entity pages will appear here after building the knowledge graph.")
        else:
            for page in entity_pages[:12]:
                with st.expander(page.stem.replace("-", " ").title(), expanded=False):
                    st.markdown(page.read_text(encoding="utf-8"))

    with stats_tab:
        stats = pipeline.get_knowledge_graph_stats()
        if stats["entity_count"] == 0:
            st.info("Build the knowledge graph to see statistics.")
        else:
            stat_cols = st.columns(4)
            stat_cols[0].metric("Entities", stats["entity_count"])
            stat_cols[1].metric("Relationships", stats["relationship_count"])
            stat_cols[2].metric("Cross-Doc Entities", stats["cross_document_entities"])
            stat_cols[3].metric("Entity Pages", stats["entity_page_count"])

            if stats.get("built_at"):
                st.caption(f"Last built: {stats['built_at'][:19].replace('T', ' ')} UTC")

            type_counts = stats.get("entity_type_counts", {})
            if type_counts:
                st.markdown("### Entity breakdown")
                for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    st.write(f"- **{etype}**: {count}")

            graph_data = pipeline.get_knowledge_graph_data()
            if graph_data.entities:
                cross = [
                    e for e in graph_data.entities
                    if len({m.get("source", "") for m in e.mentions}) > 1
                ]
                if cross:
                    st.markdown("### Cross-document entities")
                    for e in cross[:8]:
                        sources = sorted({m.get("source", "") for m in e.mentions})
                        st.write(
                            f"- **{e.name}** ({e.entity_type}): appears in {', '.join(sources)}"
                        )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    inject_styles()
    initialize_session_state()

    config = load_config()
    auth_manager = get_auth_manager()
    if not render_login(auth_manager):
        return

    workspace_name = ensure_workspace_selected(auth_manager)
    pipeline = get_pipeline(workspace_name)

    if st.session_state.conversation_id is None:
        create_new_conversation()

    render_sidebar(pipeline, config, auth_manager)
    render_header(pipeline)

    chat_tab, library_tab, wiki_tab, graph_tab = st.tabs(
        ["Chat", "Library", "Wiki", "Knowledge Graph"]
    )
    with chat_tab:
        if not st.session_state.messages:
            render_quick_prompts()
        left_col, right_col = st.columns([1.6, 1.05])
        with left_col:
            render_chat_history()
            handle_question(pipeline)
        with right_col:
            render_last_answer_summary(st.session_state.last_context)
            render_context_panel(st.session_state.last_context)

    with library_tab:
        render_library(pipeline.get_library(), pipeline)

    with wiki_tab:
        render_wiki_view(pipeline)

    with graph_tab:
        render_knowledge_graph_view(pipeline)


if __name__ == "__main__":
    main()
