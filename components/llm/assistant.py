from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import streamlit as st  # Optional: for session access if needed
except Exception:
    st = None  # Non-Streamlit environments

from openai import OpenAI


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to JSON-encode: {e}"})


def _sectionize_markdown(md: str) -> Dict[str, str]:
    """
    Very light heading-based splitter. Looks for '### ' headings.
    Returns dict: section_title -> section_content
    """
    if not md:
        return {}
    sections: Dict[str, str] = {}
    current = "Answer"
    buf: List[str] = []
    for line in md.splitlines():
        if line.strip().startswith("### "):
            # flush previous
            if buf:
                sections[current] = "\n".join(buf).strip()
                buf = []
            current = line.strip().replace("### ", "").strip()
        buf.append(line)
    if buf:
        sections[current] = "\n".join(buf).strip()
    return sections


class ToolEnabledLLM:
    """
    Wrapper around OpenAI Chat with tool calling to:
      - inspect DataFrame schema
      - compute basic stats on columns
      - generate well-structured tables
      - propose actionable calculation suggestions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_rounds: int = 4,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.max_rounds = max_rounds
        self.client = OpenAI(api_key=self.api_key)
        # Store tables the model asks to generate via tool calls
        self.generated_tables: List[Tuple[str, pd.DataFrame]] = []
        # Store calculation suggestions (textual)
        self.suggestions: List[Dict[str, Any]] = []

    def _build_tool_schemas(self, df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        # Always available tools (do not depend on df)
        tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "create_table",
                    "description": "Create a named tabular output to render in the UI. Use when you want to present structured results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Short table name"},
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Column names for the table",
                            },
                            "rows": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {},
                                },
                                "description": "List of data rows matching the column order",
                            },
                        },
                        "required": ["name", "columns", "rows"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "suggest_calculations",
                    "description": "Propose up to 5 concise, actionable calculations relevant to the user’s dataset and question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "description": {"type": "string"},
                                        "columns": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Columns involved (if known).",
                                        },
                                        "operation": {
                                            "type": "string",
                                            "description": "Plain-language description of the calculation.",
                                        },
                                    },
                                    "required": ["title", "description"],
                                },
                            }
                        },
                        "required": ["items"],
                    },
                },
            },
        ]

        # Data-aware tools (only when df is available)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            tools.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "peek_columns",
                            "description": "List available columns and inferred dtypes.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "compute_stats",
                            "description": "Compute basic statistics for specified numeric columns.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Numeric columns to summarize",
                                    },
                                    "metrics": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": [
                                                "count",
                                                "mean",
                                                "std",
                                                "min",
                                                "max",
                                                "median",
                                                "p10",
                                                "p90",
                                            ],
                                        },
                                        "description": "Metrics to compute",
                                    },
                                },
                                "required": ["columns"],
                            },
                        },
                    },
                ]
            )
        return tools

    def _handle_tool_call(
        self,
        name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame],
    ) -> str:
        """
        Execute tool call and return a JSON-string content representing the result
        to be appended as a 'tool' message.
        """
        try:
            if name == "peek_columns":
                if df is None:
                    return _safe_json_dumps({"error": "No dataset available"})
                cols = []
                for c in df.columns.tolist():
                    dtype = str(df[c].dtype)
                    cols.append({"name": c, "dtype": dtype})
                return _safe_json_dumps({"columns": cols})

            if name == "compute_stats":
                if df is None:
                    return _safe_json_dumps({"error": "No dataset available"})
                cols = arguments.get("columns", [])
                metrics = arguments.get(
                    "metrics",
                    ["count", "mean", "std", "min", "median", "max", "p10", "p90"],
                )
                numeric_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if not numeric_cols:
                    return _safe_json_dumps({"error": "No valid numeric columns in request"})

                result_rows: List[Dict[str, Any]] = []
                for c in numeric_cols:
                    series = pd.to_numeric(df[c], errors="coerce")
                    row: Dict[str, Any] = {"column": c}
                    if "count" in metrics:
                        row["count"] = int(series.count())
                    if "mean" in metrics:
                        row["mean"] = float(series.mean())
                    if "std" in metrics:
                        row["std"] = float(series.std())
                    if "min" in metrics:
                        row["min"] = float(series.min())
                    if "median" in metrics:
                        row["median"] = float(series.median())
                    if "max" in metrics:
                        row["max"] = float(series.max())
                    if "p10" in metrics:
                        row["p10"] = float(series.quantile(0.10))
                    if "p90" in metrics:
                        row["p90"] = float(series.quantile(0.90))
                    result_rows.append(row)

                # Also push a structured table for the UI
                table_df = pd.DataFrame(result_rows)
                self.generated_tables.append(("Computed Stats", table_df))
                return _safe_json_dumps({"ok": True, "rows": result_rows})

            if name == "create_table":
                # LLM proposes a named table; we store it for the UI
                tbl_name = arguments.get("name", "Table")
                columns = arguments.get("columns", [])
                rows = arguments.get("rows", [])
                try:
                    df_table = pd.DataFrame(rows, columns=columns)
                except Exception:
                    # try to coerce
                    df_table = pd.DataFrame(rows)
                    if columns and len(columns) == df_table.shape[1]:
                        df_table.columns = columns
                self.generated_tables.append((tbl_name, df_table))
                return _safe_json_dumps({"ok": True, "table": {"name": tbl_name, "columns": columns, "rows": rows}})

            if name == "suggest_calculations":
                items = arguments.get("items", [])
                # Store for UI
                if isinstance(items, list):
                    self.suggestions.extend(items)
                return _safe_json_dumps({"ok": True, "count": len(items)})
            
            if name == "query_uploaded_data":
                try:
                    # Lazy import to avoid hard dependency at import time
                    from components.rag.hybrid_retrieval import HybridRetriever
                    retriever = HybridRetriever()
                    results = retriever.retrieve_hybrid(
                        query=arguments.get("query", ""),
                        k=int(arguments.get("k", 5)),
                        sources=["data"],
                        weights={"kb": 0.0, "data": 1.0},
                    )
                    return _safe_json_dumps({"results": results})
                except Exception as e:
                    return _safe_json_dumps({"error": f"query_uploaded_data failed: {e}"})

            return _safe_json_dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return _safe_json_dumps({"error": f"Tool '{name}' failed: {e}"})


    def ask(
        self,
        prompt: str,
        system_prompt: str,
        df: Optional[pd.DataFrame] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        detail_level: str = "standard",
        enable_tools: bool = True,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "text": str,                     # model answer in markdown
            "sections": Dict[str, str],      # heading-based sections (if any)
            "tables": List[(name, DataFrame)],
            "suggestions": List[Dict]
          }
        """
        self.generated_tables.clear()
        self.suggestions.clear()

        messages: List[Dict[str, str]] = []

        # Provide dataset schema context to help the model be concrete
        schema_hint = ""
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            sample_cols = df.columns.tolist()[:50]  # cap to be safe
            schema_hint = (
                "Dataset schema (first 50 columns shown):\n"
                + ", ".join(sample_cols)
                + "\nIf you need to compute stats or produce tables, use the available tools."
            )

        detail_instruction = {
            "brief": "Be concise. Prioritize high-level insights. Use tables only if critical.",
            "standard": "Be comprehensive but focused. Use compact tables where they clarify results.",
            "deep": "Provide an in-depth analysis with step-by-step reasoning, multiple perspectives, and supporting tables where useful.",
        }.get(detail_level, "Be comprehensive but focused.")

        system = (
            f"{system_prompt}\n"
            f"- {detail_instruction}\n"
            "- When useful, structure the answer with markdown headings: '### Summary', '### Details', '### Recommendations'.\n"
            "- Only cite sources when explicitly provided in the prompt/context.\n"
        )

        if enable_tools:
            system += "- Use tools for data inspection, statistics, and emitting tabular outputs instead of fabricating numeric results.\n"
        else:
            system += "- Do not invent specific numeric values; describe the method to compute them if needed.\n"

        if schema_hint:
            system += "\n" + schema_hint

        if extra_context:
            system += "\n" + extra_context

        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        tools = self._build_tool_schemas(df) if enable_tools else None

        # Tool loop
        for _ in range(self.max_rounds):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if enable_tools else None,
                tool_choice="auto" if enable_tools else None,
            )
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)

            # If tool calls exist, handle them
            tool_calls = getattr(choice.message, "tool_calls", None)
            if enable_tools and tool_calls:
                # Append assistant message with tool_calls as-is
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [tc.dict() if hasattr(tc, "dict") else dict(tc) for tc in tool_calls],
                    }
                )
                # Execute each tool call
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        arguments = {}
                    tool_result = self._handle_tool_call(name, arguments, df)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": tool_result,
                        }
                    )
                # Continue the loop to let the model observe the tool outputs
                continue

            # No tool calls: finalize
            text = choice.message.content or ""
            sections = _sectionize_markdown(text)
            return {
                "text": text,
                "sections": sections,
                "tables": list(self.generated_tables),
                "suggestions": list(self.suggestions),
            }

        # If loop exits without a normal return
        return {
            "text": "The assistant reached the maximum tool-calling steps without concluding.",
            "sections": {},
            "tables": list(self.generated_tables),
            "suggestions": list(self.suggestions),
        }