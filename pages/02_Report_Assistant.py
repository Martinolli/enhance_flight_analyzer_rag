# New flexible, expandable responses with data-aware tools

import os
import re
import textwrap
import tempfile

import pandas as pd
import streamlit as st

# Set page config early
st.set_page_config(page_title="Knowledge & Report Assistant", page_icon="🧭", layout="wide")

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from components.rag.bootstrap import ensure_rag_db
from components.llm.assistant import ToolEnabledLLM
from components.data_ingest import DataIngestor
from components.rag.hybrid_retrieval import HybridRetriever

# --- SESSION STATE INITIALIZATION ---
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "assistant" not in st.session_state:
    # Optional: prefer st.secrets["OPENAI_API_KEY"] if available
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    except Exception:
        # Fallback if no secrets.toml file exists
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1")
    st.session_state.assistant = ToolEnabledLLM(api_key=OPENAI_API_KEY, model=MODEL_NAME)

# --- UI & HELPER FUNCTIONS ---
st.title("🧭 Knowledge & Report Assistant")
st.caption("Your conversational partner for flight data analysis, knowledge retrieval, and report generation.")

LATEX_BLOCK_PATTERN = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)

def render_markdown_with_latex(content: str, target=None) -> None:
    """Render markdown content with block-level LaTeX support."""
    if not content:
        return
    if target is None:
        target = st

    markdown_fn = getattr(target, "markdown", None)
    if markdown_fn is None:
        return
    latex_fn = getattr(target, "latex", None)

    cursor = 0
    for match in LATEX_BLOCK_PATTERN.finditer(content):
        before = content[cursor:match.start()]
        if before.strip():
            markdown_fn(before)

        expr = match.group(1).strip()
        if expr:
            if callable(latex_fn):
                latex_fn(expr)
            else:
                markdown_fn(f"$$\n{expr}\n$$")
        cursor = match.end()

    tail = content[cursor:]
    if tail.strip():
        markdown_fn(tail)

def render_chat_messages():
    """Render the chat history."""
    for msg in st.session_state.assistant_messages:
        with st.chat_message(msg["role"]):
            # Render text content
            render_markdown_with_latex(msg["content"])
            
            # Render tables if they exist in the message
            if "tables" in msg and msg["tables"]:
                for tname, tdf in msg["tables"]:
                    with st.expander(f"📊 {tname}", expanded=False):
                        st.dataframe(tdf, use_container_width=True)
                        csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"Download {tname}",
                            data=csv_bytes,
                            file_name=f"{tname.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                        )
            
            # Render suggestions if they exist
            if "suggestions" in msg and msg["suggestions"]:
                st.markdown("#### 🤔 Suggested Next Steps:")
                for s in msg["suggestions"]:
                    st.markdown(f"- **{s.get('title','Suggestion')}**: {s.get('description','')}")
            
            # NEW: render figures if present
            if "figures" in msg and msg["figures"]:
                st.markdown("#### 📈 Charts")
                for title, fig in msg["figures"]:
                    if title:
                        st.caption(title)
                    st.pyplot(fig, clear_figure=False)
            
            # Render citations (sources)
            if "sources" in msg and msg["sources"]:
                st.markdown("---")
                with st.expander("📚 Citations", expanded=False):
                    for i, h in enumerate(msg["sources"], 1):
                        meta = h.get("metadata", {})
                        source_name = meta.get("source") or meta.get("path") or meta.get("file_name", "Unknown Source")
                        st.caption(f"[{i}] {source_name}")
                        st.write(h.get("text", "")[:1000] + "...")

@st.cache_data(ttl=300, show_spinner=False)
def cached_retrieve(query: str, k: int, db_path: str, sources: tuple[str, ...], file_id: str | None = None):
    """Cached retrieval function."""
    retriever = HybridRetriever(db_path=db_path)
    if not sources:
        return []
    
    weights = {"kb": 0.5, "data": 0.5}
    if "kb" not in sources:
        weights["data"] = 1.0
    if "data" not in sources:
        weights["kb"] = 1.0

    return retriever.retrieve_hybrid(
        query, k=k, sources=list(sources), weights=weights, file_id=file_id
    )

# --- SIDEBAR: CONTROLS & SETTINGS ---
with st.sidebar:
    st.header("⚙️ Assistant Settings")

    st.subheader("📁 Data Context")
    uploaded_file = st.file_uploader("Upload CSV/XLSX/Parquet to embed", type=["csv", "xlsx", "parquet"])
    if uploaded_file:
        # Save to a cross-platform temp file
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name
        
        # Ingest
        ingestor = DataIngestor()
        with st.spinner("Processing and embedding data file..."):
            result = ingestor.ingest_file(temp_path, uploaded_file.name)
        
        os.remove(temp_path)
        
        if result["status"] == "success":
            st.success(f"✅ Ingested '{uploaded_file.name}' ({result['row_count']} rows)")
        else:
            st.error(f"❌ {result['message']}")

    st.subheader("🧠 Knowledge Retrieval (RAG)")
    enable_rag = st.checkbox("Enable RAG", value=True, help="Allow assistant to retrieve from knowledge base.")
    if enable_rag:
        db_path = st.text_input("Vector DB path", value=".ragdb")
        top_k = st.slider("Top-K passages", 1, 12, 5)
        include_data_context = st.checkbox("Include data embeddings", value=True, help="Also search in the uploaded data file's embeddings.")
    
    st.subheader("🤖 Model & Response")
    detail_level = st.radio(
        "Detail level", ["brief", "standard", "deep"], index=1, horizontal=True,
        help="Controls how comprehensive the answer is."
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)


# --- MAIN CHAT INTERFACE ---
render_chat_messages()

prompt = st.chat_input("Ask about your data, the knowledge base, or request a report...")

if prompt:
    # Add user message to history
    st.session_state.assistant_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- PREPARE & CALL THE ASSISTANT ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # Get current dataset from main app page
            df = st.session_state.get("data")
            
            # Prepare context from RAG
            rag_context = ""
            hits = []
            if enable_rag and ensure_rag_db(db_path=db_path):
                sources_to_query = ["kb"]
                if include_data_context:
                    sources_to_query.append("data")
                
                with st.spinner("Retrieving context..."):
                    hits = cached_retrieve(prompt, top_k, db_path, tuple(sources_to_query))

                if hits:
                    kb_hits = [h for h in hits if h.get("source_type") == "kb"]
                    data_hits = [h for h in hits if h.get("source_type") == "data"]
                    
                    if kb_hits:
                        context_kb = "\n\n".join([f"[KB {i+1}] {h['text']}" for i, h in enumerate(kb_hits)])
                        rag_context += f"\n\nKnowledge Base Sources:\n{context_kb}\n"
                    if data_hits:
                        context_data = "\n\n".join([f"[DATA {i+1}] {h['text']}" for i, h in enumerate(data_hits)])
                        rag_context += f"\n\nUploaded Data Context (semantic matches):\n{context_data}\n"

            # >>> FIX: Truncate rag_context to avoid exceeding token limits
            # A conservative limit to leave room for history, prompt, and response.
            # 24000 characters is roughly 6000 tokens.
            MAX_CONTEXT_CHARS = 24000
            if len(rag_context) > MAX_CONTEXT_CHARS:
                rag_context = rag_context[:MAX_CONTEXT_CHARS] + "\n\n...[CONTEXT TRUNCATED]..."


            # Prepare the system prompt
            system_prompt = textwrap.dedent("""
            You are a world-class Flight Test engineering assistant. 
            - Ground your answers in the provided sources, citing them as [KB N] or [DATA N].
            - If sources are insufficient, state what is missing and use your general knowledge.
            - When asked to analyze data, use the available tools to inspect the dataset, compute statistics, and generate tables.
            - For mathematical formulas, use LaTeX syntax enclosed in `$` or `$$` delimiters (e.g., `$\\Delta V / \\Delta t$`).
            - Be concise, technical, and helpful.
            """).strip()

            # Get the existing chat history for context
            # We need to filter out our custom keys ('tables', 'suggestions', etc.) before sending to OpenAI
            openai_history = []
            for msg in st.session_state.assistant_messages:
                openai_history.append({k: v for k, v in msg.items() if k in ["role", "content", "tool_calls"]})

            # Call the assistant
            assistant = st.session_state.assistant
            result = assistant.ask(
                prompt=prompt,
                system_prompt=system_prompt,
                df=df,
                chat_history=openai_history[:-1], # Pass history *before* the current user prompt
                temperature=temperature,
                max_tokens=4000,
                detail_level=detail_level,
                enable_tools=True,
                extra_context=rag_context,
            )

            # Update the placeholder with the final response
            message_placeholder.empty()
            render_markdown_with_latex(result["text"], target=message_placeholder)
            
            # --- STORE THE FULL RESPONSE ---
            # The last message in the history is the user's prompt, so we pop it
            st.session_state.assistant_messages.pop() 
            
            # Add user prompt back, but this time just as a simple dict
            st.session_state.assistant_messages.append({"role": "user", "content": prompt})

            # Add the final assistant response with all the rendered components
            final_assistant_message = {
                "role": "assistant",
                "content": result["text"],
                "tables": result["tables"],
                "suggestions": result["suggestions"],
                "sources": hits,
                "figures": result.get("figures", []),  # NEW
            }
            st.session_state.assistant_messages.append(final_assistant_message)
            
            # We also need to store the raw OpenAI history for the next turn
            # The `ask` method now returns the history, so we can update our session state with it
            # This is a bit complex: we need to merge our rich history with the raw OpenAI history
            # For simplicity, we will just rebuild the chat display from our rich history
            # and let the LLM `ask` method manage the raw history.
            # So, the `openai_history` sent to `ask` is correct.

            # Re-render the new elements in the final message placeholder
            if result["tables"]:
                for tname, tdf in result["tables"]:
                    with st.expander(f"📊 {tname}", expanded=True):
                        st.dataframe(tdf, use_container_width=True)
                        csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"Download {tname}",
                            data=csv_bytes,
                            file_name=f"{tname.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                        )

            if result["suggestions"]:
                st.markdown("#### 🤔 Suggested Next Steps:")
                for s in result["suggestions"]:
                    st.markdown(f"- **{s.get('title','Suggestion')}**: {s.get('description','')}")
            
            if hits:
                with st.expander("📚 Citations", expanded=False):
                    # We need a function to render sources here since this is outside the main render loop
                    for i, h in enumerate(hits, 1):
                        meta = h.get("metadata", {})
                        source_name = meta.get("source") or meta.get("path") or meta.get("file_name", "Unknown Source")
                        st.caption(f"[{i}] {source_name}")
                        st.write(h.get("text", "")[:1000] + "...")

            # NEW: show figures immediately in this turn
            if result.get("figures"):
                st.markdown("#### 📈 Charts")
                for title, fig in result["figures"]:
                    if title:
                        st.caption(title)
                    st.pyplot(fig, clear_figure=False)


st.divider()
with st.expander("Initialize knowledge base (if missing)"):
    init_db_path = st.text_input("DB path", value=".ragdb", key="init_db_path_2")
    if st.button("Ensure KB now"):
        ok = ensure_rag_db(db_path=init_db_path)
        if ok:
            st.success(f"Knowledge base is ready at {init_db_path}.")
        else:
            st.warning("KB not ready. Set RAG_DB_ZIP_URL in secrets or include docs/knowledge_base and try again.")
