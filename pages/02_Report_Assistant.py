# New flexible, expandable responses with data-aware tools

import os
import textwrap
from datetime import timedelta
import tempfile

import pandas as pd
import streamlit as st

# Set page config early
st.set_page_config(page_title="Knowledge & Report Assistant", page_icon="🧭", layout="wide")

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

from components.rag.retrieval import retrieve
from components.rag.bootstrap import ensure_rag_db
from components.llm.assistant import ToolEnabledLLM  # NEW
from components.data_ingest import DataIngestor

st.subheader("📁 Upload Flight Data")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "parquet"])

if uploaded_file:
    # Save to a cross-platform temp file (Windows safe)
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
    
    # Ingest
    ingestor = DataIngestor()
    with st.spinner("Processing and embedding..."):
        result = ingestor.ingest_file(temp_path, uploaded_file.name)
    
    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass
    
    if result["status"] == "success":
        st.success(f"✅ Ingested {result['row_count']} rows")
    else:
        st.error(f"❌ {result['message']}")


# Optional: prefer st.secrets["OPENAI_API_KEY"] if available
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
except Exception:
    # Fallback if no secrets.toml file exists
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4-turbo")  # change as desired

st.set_page_config(page_title="Knowledge & Report Assistant", page_icon="🧭", layout="wide")
st.title("🧭 Knowledge & Report Assistant")

# Helper to summarize current dataset
def dataset_summary(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No dataset loaded."
    n_rows, n_cols = df.shape
    head_cols = ", ".join(df.columns[:12]) + ("..." if n_cols > 12 else "")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return textwrap.dedent(
        f"""Rows: {n_rows:,}, Columns: {n_cols}
Numeric columns (sample): {", ".join(num_cols[:8]) + ("..." if len(num_cols) > 8 else "")}
First columns: {head_cols}
"""
    )

def render_sources(sources):
    if not sources:
        st.info("No sources used.")
        return
    for i, h in enumerate(sources, 1):
        with st.expander(f"Source {i} ({h.get('source', 'KB')})", expanded=False):
            st.caption(h.get("metadata", {}).get("path", ""))
            st.write(h.get("text", "")[:2000])

def call_llm(
    prompt: str,
    temperature: float = 0.6,
    max_tokens: int = 4000,
    system_prompt: str = "You are a Flight Test engineering assistant. Be precise and cite sources when provided.",
) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set. Please configure in Streamlit secrets or environment."
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# Left: KB query; Right: Report generation
left, right = st.columns([0.48, 0.52])

with left:
    st.subheader("Knowledge Base Q&A (RAG)")
    st.caption("Ask a question. The assistant retrieves context from your local knowledge base and answers with citations.")
    query = st.text_input("Question", placeholder="e.g., How do we interpret ITT spikes during transient maneuvers?")
    scope = st.selectbox(
        "Answer scope",
        ["Prefer KB (default)", "KB + Model knowledge", "Model-only"],
        index=0,
        help="Choose whether to ground the answer in the KB only, combine KB with the model's general knowledge, or ignore the KB and use the model only.",
    )
    top_k = st.slider("Top-K passages", 1, 12, 6)

    # NEW: Flexible response controls
    st.markdown("#### Response controls")
    detail_level = st.radio(
        "Detail level",
        ["brief", "standard", "deep"],
        index=1,
        horizontal=True,
        help="Controls how comprehensive the answer is.",
    )
    enable_tools = st.checkbox(
        "Enable smart tools (tables & calculation suggestions)",
        value=True,
        help="Allow the model to compute stats and emit tables from your uploaded dataset.",
    )
    db_path = st.text_input("Vector DB path", value=".ragdb")
    fallback_to_model = st.checkbox(
        "Fallback to model-only if KB has no hits",
        value=True,
        help="If no passages are retrieved from the KB, answer using the model's general knowledge.",
    )

    if st.button("Search & Answer", use_container_width=True, type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            df = st.session_state.get("data")
            if scope == "Model-only":
                with st.spinner("Thinking (model-only)..."):
                    system_prompt = (
                        "You are a Flight Test engineering assistant. Use your domain knowledge. "
                        "Do not fabricate citations. If uncertain, say what additional data would help."
                    )
                    if enable_tools:
                        # Data-aware multi-tool path
                        assistant = ToolEnabledLLM(api_key=OPENAI_API_KEY, model=MODEL_NAME)
                        result = assistant.ask(
                            prompt=query,
                            system_prompt=system_prompt,
                            df=df,
                            temperature=0.6,
                            max_tokens=4000,
                            detail_level=detail_level,
                            enable_tools=True,
                        )
                        # Render expandable response
                        with st.expander("Answer", expanded=True):
                            st.markdown(result["text"])
                        if result["tables"]:
                            st.markdown("#### Generated Tables")
                            for tname, tdf in result["tables"]:
                                with st.expander(f"{tname}", expanded=False):
                                    st.dataframe(tdf, use_container_width=True)
                                    csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                                    st.download_button(
                                        f"Download {tname} (CSV)",
                                        data=csv_bytes,
                                        file_name=f"{tname.lower().replace(' ', '_')}.csv",
                                        mime="text/csv",
                                    )
                        if result["suggestions"]:
                            st.markdown("#### Suggested Calculations")
                            for i, s in enumerate(result["suggestions"], 1):
                                st.markdown(f"- {i}. **{s.get('title','Suggestion')}** — {s.get('description','')}")
                    else:
                        # Fallback simple model call
                        answer = call_llm(query, system_prompt=system_prompt)
                        with st.expander("Answer", expanded=True):
                            st.write(answer)
            else:
                if not ensure_rag_db(db_path=db_path):
                    st.error("Knowledge base not available. Initialize it below or check your path.")
                else:
                    with st.spinner("Retrieving from KB..."):
                        hits = retrieve(query, k=top_k, db_path=db_path)

                    if not hits and not fallback_to_model:
                        st.info("No KB hits found. Enable fallback to answer from general knowledge or refine your query.")
                    else:
                        # Compose prompt depending on scope
                        context = "\n\n".join([f"[Source {i+1}] {h['text']}" for i, h in enumerate(hits)])

                        if hits:
                            kb_section = f"\n\nSources:\n{context}\n"
                        else:
                            kb_section = ""

                        if scope == "Prefer KB (default)":
                            prompt = f"""Ground your answer primarily in the provided sources. Cite as [Source N]. If the sources are insufficient, say what is missing and answer briefly using general knowledge.
Question: {query}
{kb_section}
Answer:"""
                            system_prompt = "You are a Flight Test engineering assistant. Be precise and cite sources when provided."
                        else:  # KB + Model knowledge
                            prompt = f"""Use both the provided sources and your broader Flight Test knowledge. When a statement is directly supported by a source, cite it as [Source N]. For domain knowledge not in the sources, include it without a [Source] tag. If sources conflict, note the discrepancy.
Question: {query}
{kb_section}
Answer:"""
                            system_prompt = "You are a Flight Test engineering assistant. Be precise and cite sources when provided."

                        # RAG answer (tool-enabled if requested)
                        if enable_tools:
                            assistant = ToolEnabledLLM(api_key=OPENAI_API_KEY, model=MODEL_NAME)
                            result = assistant.ask(
                                prompt=prompt,
                                system_prompt=system_prompt,
                                df=st.session_state.get("data"),
                                temperature=0.6,
                                max_tokens=4000,
                                detail_level=detail_level,
                                enable_tools=True,
                            )
                            with st.expander("Answer", expanded=True):
                                st.markdown(result["text"])
                            if hits:
                                st.markdown("**Citations:**")
                                render_sources(hits)
                            if result["tables"]:
                                st.markdown("#### Generated Tables")
                                for tname, tdf in result["tables"]:
                                    with st.expander(f"{tname}", expanded=False):
                                        st.dataframe(tdf, use_container_width=True)
                                        csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                                        st.download_button(
                                            f"Download {tname} (CSV)",
                                            data=csv_bytes,
                                            file_name=f"{tname.lower().replace(' ', '_')}.csv",
                                            mime="text/csv",
                                        )
                            if result["suggestions"]:
                                st.markdown("#### Suggested Calculations")
                                for i, s in enumerate(result["suggestions"], 1):
                                    st.markdown(f"- {i}. **{s.get('title','Suggestion')}** — {s.get('description','')}")
                        else:
                            with st.spinner("Thinking..."):
                                answer = call_llm(prompt)
                            with st.expander("Answer", expanded=True):
                                st.markdown(answer)
                            if hits:
                                st.markdown("**Citations:**")
                                render_sources(hits)

with right:
    st.subheader("Generate Flight Test Report from Current Dataset")
    df = st.session_state.get("data")
    if df is None or df.empty:
        st.info("Upload data on the main page to enable report generation.")
    else:
        st.text_area("Dataset Summary (auto-generated)", value=dataset_summary(df), height=160, key="auto_ds_sum")
        goal = st.text_input(
            "Report goal / scenario",
            value="Provide a concise engineering report focusing on torque (PMU ENGINE TORQUE), ITT, and NP behavior over the selected timeframe, describing anomalies and recommendations.",
        )
        include_tables = st.checkbox("Include brief stats table", value=True)
        rag_k = st.slider("RAG Top-K", 2, 12, 6, help="How many knowledge passages to include as context")
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
        db_path2 = st.text_input("Vector DB path (report)", value=".ragdb", key="db2")
        enable_tools_r = st.checkbox("Enable smart tools for report (tables & stats)", value=True)
        detail_level_r = st.radio("Report detail", ["brief", "standard", "deep"], index=1, horizontal=True)

        if st.button("Generate Report", use_container_width=True):
            hits = []
            background_section = ""
            if ensure_rag_db(db_path=db_path2):
                with st.spinner("Retrieving report context from KB..."):
                    hits = retrieve(goal, k=rag_k, db_path=db_path2)

            if hits:
                background_section = "Knowledge Base Context:\n" + "\n\n".join(
                    [f"[KB {i+1}] {h['text']}" for i, h in enumerate(hits)]
                )

            prompt = f"""You are to write a Flight Test engineering report based on the current dataset and the stated goal.

Goal:
{goal}

If a timeframe was selected on the main page, focus your narrative on that interval.
{background_section}
Format:
1. Executive Summary (<= 1000 words)
2. Data Overview (sampling, parameters monitored, timeframe)
3. Findings and Trends (quantitative; refer to signals by their names)
4. Anomalies and Events (possible causes and implications)
5. Recommendations (next steps, additional tests)
6. Limitations and Assumptions
7. References (list [KB N] items you used if applicable)

Be concise, technical, and specific to the dataset.
"""
            with st.spinner("Generating report..."):
                system_prompt = (
                    "You are a Flight Test engineering assistant. Be precise and cite sources when provided. "
                    "If no sources are provided, rely on domain knowledge without fabricating citations."
                )

                if enable_tools_r:
                    assistant = ToolEnabledLLM(api_key=OPENAI_API_KEY, model=MODEL_NAME)
                    result = assistant.ask(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        df=df,
                        temperature=temperature,
                        max_tokens=4000,
                        detail_level=detail_level_r,
                        enable_tools=True,
                        extra_context="When producing summary statistics or compact parameter tables, prefer the 'create_table' and 'compute_stats' tools.",
                    )
                    st.markdown("### Report")
                    with st.expander("Full report", expanded=True):
                        st.markdown(result["text"])
                    if result["tables"]:
                        st.markdown("### Report Tables")
                        for tname, tdf in result["tables"]:
                            with st.expander(f"{tname}", expanded=False):
                                st.dataframe(tdf, use_container_width=True)
                                csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    f"Download {tname} (CSV)",
                                    data=csv_bytes,
                                    file_name=f"{tname.lower().replace(' ', '_')}.csv",
                                    mime="text/csv",
                                )
                else:
                    report = call_llm(prompt, temperature=temperature, max_tokens=1400, system_prompt=system_prompt)
                    st.markdown("### Report")
                    with st.expander("Full report", expanded=True):
                        st.write(report)
            if hits:
                st.markdown("**Citations:**")
                render_sources(hits)

st.divider()
st.caption(
    "Tip: Provide a prebuilt KB via secrets RAG_DB_ZIP_URL, or build locally by running the ingestion script on your PDFs/TXT and setting OPENAI_API_KEY (or install sentence-transformers for local embeddings)."
)

# Optional one-click initializer in hosted environments
with st.expander("Initialize knowledge base (if missing)"):
    default_db = ".ragdb"
    init_db_path = st.text_input("DB path", value=default_db, key="init_db_path")
    if st.button("Ensure KB now"):
        ok = ensure_rag_db(db_path=init_db_path)
        if ok:
            st.success(f"Knowledge base is ready at {init_db_path}.")
        else:
            st.warning("KB not ready. Set RAG_DB_ZIP_URL in secrets or include docs/knowledge_base and try again.")
