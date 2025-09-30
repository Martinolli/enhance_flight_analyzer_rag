import os
import textwrap
from datetime import timedelta

import pandas as pd
import streamlit as st

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from components.rag.retrieval import retrieve
from components.rag.bootstrap import ensure_rag_db

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
    lines = []
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Time info (if available)
    if "Elapsed Time (s)" in df.columns and df["Elapsed Time (s)"].notna().any():
        tmin = float(df["Elapsed Time (s)"].min())
        tmax = float(df["Elapsed Time (s)"].max())
        dt = tmax - tmin
        lines.append(f"Elapsed time range: {tmin:.3f} s to {tmax:.3f} s (duration {dt:.3f} s).")
        # Median sampling
        diffs = pd.Series(df["Elapsed Time (s)"]).diff().dropna()
        if not diffs.empty and diffs.median() > 0:
            fs = 1.0 / diffs.median()
            lines.append(f"Estimated sampling rate: {fs:.2f} Hz.")
    elif "Timestamp" in df.columns and df["Timestamp"].notna().any():
        tmin = pd.to_datetime(df["Timestamp"], errors="coerce").min()
        tmax = pd.to_datetime(df["Timestamp"], errors="coerce").max()
        if pd.notna(tmin) and pd.notna(tmax):
            dt = (tmax - tmin).total_seconds()
            lines.append(f"Timestamp range: {tmin} to {tmax} (duration {dt:.3f} s).")

    # List up to 12 columns (skip Timestamp text columns)
    cols = [c for c in df.columns if c not in ("Timestamp",)]
    lines.append("Sample columns: " + ", ".join(cols[:12]) + (" ..." if len(cols) > 12 else ""))

    return "\n".join(lines)

def render_sources(sources):
    for s in sources:
        meta = s.get("metadata", {})
        src = meta.get("source", "unknown")
        page = meta.get("page")
        st.markdown(f"- {src}" + (f", page {page}" if page else ""))

def call_llm(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
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
            # Handle Model-only early to skip RAG entirely
            if scope == "Model-only":
                with st.spinner("Thinking (model-only)..."):
                    system_prompt = (
                        "You are a Flight Test engineering assistant. Use your domain knowledge. "
                        "Do not fabricate citations. If uncertain, say what additional data would help."
                    )
                    answer = call_llm(query, system_prompt=system_prompt)
                st.markdown("**Answer:**")
                st.write(answer)
            else:
                if not ensure_rag_db(db_path=db_path):
                    st.error("Knowledge base not available. Configure RAG_DB_ZIP_URL in secrets or ingest docs locally.")
                else:
                    try:
                        hits = retrieve(query=query, k=top_k, db_path=db_path)
                    except Exception as e:
                        st.error(f"Retrieval failed: {e}. Ensure your deployed embedding backend matches the DB (e.g., OPENAI_API_KEY set if the DB was built with OpenAI).")
                        hits = []

                    if not hits:
                        if fallback_to_model:
                            st.info("No results in the KB. Falling back to model-only answer.")
                            with st.spinner("Thinking (model-only)..."):
                                system_prompt = (
                                    "You are a Flight Test engineering assistant. Use your domain knowledge. "
                                    "Do not fabricate citations. If uncertain, say what additional data would help."
                                )
                                answer = call_llm(query, system_prompt=system_prompt)
                            st.markdown("**Answer:**")
                            st.write(answer)
                        else:
                            st.info("No results in the KB. Ingest documents first or enable fallback.")
                    else:
                        # Compose prompt depending on scope
                        context = "\n\n".join([f"[Source {i+1}] {h['text']}" for i, h in enumerate(hits)])

                        if scope == "Prefer KB (default)":
                            prompt = f"""Ground your answer primarily in the provided sources. Cite as [Source N]. If the sources are insufficient, say what is missing and answer briefly using general knowledge.
Question: {query}

Sources:
{context}

Answer:"""
                        else:  # KB + Model knowledge
                            prompt = f"""Use both the provided sources and your broader Flight Test knowledge. When a statement is directly supported by a source, cite it as [Source N]. For domain knowledge not in the sources, include it without a [Source] tag. If sources conflict, note the discrepancy.
Question: {query}

Sources:
{context}

Answer:"""

                        with st.spinner("Thinking..."):
                            answer = call_llm(prompt)
                        st.markdown("**Answer:**")
                        st.write(answer)
                        st.markdown("**Citations:**")
                        render_sources(hits)

with right:
    st.subheader("Generate Flight Test Report from Current Dataset")
    df = st.session_state.get("data")
    if df is None or df.empty:
        st.info("Upload data on the main page to enable report generation.")
    else:
        st.text_area("Dataset Summary (auto-generated)", value=dataset_summary(df), height=160, key="auto_ds_sum")
        goal = st.text_input("Report goal / scenario", value="Provide a concise engineering report focusing on torque (PMU ENGINE TORQUE), ITT, and NP behavior over the selected timeframe, describing anomalies and recommendations.")
        include_tables = st.checkbox("Include brief stats table", value=True)
        rag_k = st.slider("RAG Top-K", 2, 12, 6, help="How many knowledge passages to include as context")
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
        db_path2 = st.text_input("Vector DB path (report)", value=".ragdb", key="db2")
        report_scope = st.selectbox(
            "Knowledge use for background",
            ["Prefer KB (default)", "KB + Model knowledge", "Model-only"],
            index=0,
            help="Select how much to rely on the KB versus the model's general knowledge when writing the report.",
        )
        report_fallback = st.checkbox(
            "Fallback to model-only if KB has no hits (report)",
            value=True,
        )

        if st.button("Generate Report", use_container_width=True, type="primary"):
            # Build a short stats section (optional)
            stats_block = ""
            if include_tables:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                head = df[numeric_cols].describe().T[["mean", "std", "min", "max"]].round(3)
                try:
                    stats_block = head.head(12).to_markdown(index=True)
                except ImportError:
                    # Fallback if tabulate is not available
                    stats_block = head.head(12).to_string()

            # Retrieve background knowledge unless in Model-only
            kb_query = "flight test data analysis methods, interpretation of torque/ITT/NP trends, vibration analysis, and reporting templates"
            hits = []
            context = ""
            if report_scope != "Model-only":
                if not ensure_rag_db(db_path=db_path2):
                    st.error("Knowledge base not available. Configure RAG_DB_ZIP_URL in secrets or ingest docs locally.")
                else:
                    try:
                        hits = retrieve(kb_query, k=rag_k, db_path=db_path2)
                    except Exception as e:
                        st.error(f"Retrieval failed: {e}. Ensure your deployed embedding backend matches the DB (e.g., OPENAI_API_KEY set if the DB was built with OpenAI).")
                        hits = []
                if hits:
                    context = "\n\n".join([f"[KB {i+1}] {h['text']}" for i, h in enumerate(hits)])

            # Construct report prompt
            ds_text = st.session_state.get("auto_ds_sum", "")
            # Construct report prompt (adjust guidance based on scope)
            background_section = ""
            if context:
                if report_scope == "Prefer KB (default)":
                    background_intro = (
                        "Ground your background primarily in the provided KB excerpts. Cite as [KB N]. "
                        "If the KB is insufficient, say what is missing and add brief context using general knowledge."
                    )
                else:  # KB + Model knowledge
                    background_intro = (
                        "Use both KB excerpts and your broader Flight Test knowledge. Cite [KB N] when directly supported; "
                        "general knowledge can be included without [KB] tags. Note any contradictions."
                    )
                background_section = f"\nBackground knowledge (guidance):\n{background_intro}\n\n{context}\n"
            elif report_scope == "Model-only" or report_fallback:
                background_section = (
                    "\nBackground knowledge: You may use your broader Flight Test engineering knowledge. "
                    "Do not fabricate citations.\n"
                )

            prompt = f"""
Generate a structured Flight Test Engineering report.

Objectives:
- {goal}

Dataset summary:
{ds_text}

If a timeframe was selected on the main page, focus your narrative on that interval.
{background_section}
Format:
1. Executive Summary (<= 150 words)
2. Data Overview (sampling, parameters monitored, timeframe)
3. Findings and Trends (quantitative; refer to signals by their names)
4. Anomalies and Events (possible causes and implications)
5. Recommendations (next steps, additional tests)
6. Limitations and Assumptions
7. References (list [KB N] items you used if applicable)

Be concise, technical, and specific to the dataset.
"""
            with st.spinner("Generating report..."):
                # If no KB context and not explicitly Model-only, optionally fall back
                system_prompt = (
                    "You are a Flight Test engineering assistant. Be precise and cite sources when provided. "
                    "If no sources are provided, rely on domain knowledge without fabricating citations."
                )
                report = call_llm(prompt, temperature=temperature, max_tokens=1400, system_prompt=system_prompt)
            st.markdown("### Report")
            st.write(report)
            if hits:
                st.markdown("**Citations:**")
                render_sources(hits)

st.divider()
st.caption("Tip: Provide a prebuilt KB via secrets RAG_DB_ZIP_URL, or build locally by running the ingestion script on your PDFs/TXT and setting OPENAI_API_KEY (or install sentence-transformers for local embeddings).")

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