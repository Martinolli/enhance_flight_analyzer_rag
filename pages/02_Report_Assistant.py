import os
import textwrap
from datetime import timedelta

import pandas as pd
import streamlit as st

from components.rag.retrieval import retrieve

# Optional: prefer st.secrets["OPENAI_API_KEY"]
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change as desired

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

def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 1200) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set. Please configure in Streamlit secrets or environment."
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a Flight Test engineering assistant. Be precise and cite sources when provided."},
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
    top_k = st.slider("Top-K passages", 1, 12, 6)
    db_path = st.text_input("Vector DB path", value=".ragdb")
    if st.button("Search & Answer", use_container_width=True, type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            hits = retrieve(query=query, k=top_k, db_path=db_path)
            if not hits:
                st.info("No results in the KB. Ingest documents first.")
            else:
                # Compose RAG prompt
                context = "\n\n".join([f"[Source {i+1}] {h['text']}" for i, h in enumerate(hits)])
                prompt = f"""Use the following sources to answer the question. Cite as [Source N].
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

        if st.button("Generate Report", use_container_width=True, type="primary"):
            # Build a short stats section (optional)
            stats_block = ""
            if include_tables:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                head = df[numeric_cols].describe().T[["mean", "std", "min", "max"]].round(3)
                stats_block = head.head(12).to_markdown(index=True)

            # Retrieve background knowledge
            kb_query = "flight test data analysis methods, interpretation of torque/ITT/NP trends, vibration analysis, and reporting templates"
            hits = retrieve(kb_query, k=rag_k, db_path=db_path2)
            context = "\n\n".join([f"[KB {i+1}] {h['text']}" for i, h in enumerate(hits)])

            # Construct report prompt
            ds_text = st.session_state.get("auto_ds_sum", "")
            prompt = f"""
Generate a structured Flight Test Engineering report.

Objectives:
- {goal}

Dataset summary:
{ds_text}

If a timeframe was selected on the main page, focus your narrative on that interval.

Background knowledge (use as guidance; cite as [KB N] when relevant):
{context}

Format:
1. Executive Summary (<= 150 words)
2. Data Overview (sampling, parameters monitored, timeframe)
3. Findings and Trends (quantitative; refer to signals by their names)
4. Anomalies and Events (possible causes and implications)
5. Recommendations (next steps, additional tests)
6. Limitations and Assumptions
7. References (list [KB N] items you used)

Be concise, technical, and specific to the dataset.
"""
            with st.spinner("Generating report..."):
                report = call_llm(prompt, temperature=temperature, max_tokens=1400)
            st.markdown("### Report")
            st.write(report)
            st.markdown("**Citations:**")
            render_sources(hits)

st.divider()
st.caption("Tip: Build the knowledge base by running the ingestion script on your PDFs/TXT and setting OPENAI_API_KEY (or install sentence-transformers for local embeddings).")