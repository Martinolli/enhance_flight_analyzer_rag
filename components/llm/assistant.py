"""
LLM assistant with tool calling for data analysis, including anomaly, trend, and correlation analysis.
    Wraps OpenAI Chat with tools to inspect DataFrame schema, compute stats, generate tables, suggest calculations,
    and perform advanced analyses using FlightDataStatistics.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for environments without display
import matplotlib.pyplot as plt

import pandas as pd

try:
    import streamlit as st  # Optional: for session access if needed
except Exception:
    st = None  # Non-Streamlit environments

from openai import OpenAI

# NEW: Stats engine
from components.statistical_analysis import FlightDataStatistics
# NEW: Simple NL query analyzer
from components.query_understanding import QueryAnalyzer


def _safe_json_dumps(obj: Any) -> str:
    """
    Safely JSON-encode an object, returning an error message on failure.
    input: Any serializable object
    output: JSON string    
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to JSON-encode: {e}"})


def _sectionize_markdown(md: str) -> Dict[str, str]:
    """
    Very light heading-based splitter. Looks for '### ' headings.
    Returns dict: section_title -> section_content
    input: markdown string
    output: dict of sections
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


# NEW: DDD:HH:MM:SS(.mmm) flight-time parser
_TIME_RE = re.compile(
    r"^\s*(?P<d>\d{1,3}):(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<ms>\d{1,3}))?\s*$"
)

def _parse_flight_time_index(s: str) -> Optional[float]:
    """
    Parse DDD:HH:MM:SS(.mmm) into total seconds (float).
    Returns None if invalid.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    m = _TIME_RE.match(s.strip())
    if not m:
        return None
    d = int(m.group("d"))
    h = int(m.group("h"))
    mnt = int(m.group("m"))
    sec = int(m.group("s"))
    ms = int(m.group("ms") or 0)
    total_seconds = d * 86400 + h * 3600 + mnt * 60 + sec + ms / 1000.0
    return float(total_seconds)


def _apply_time_window_filter(
    df: pd.DataFrame,
    time_window: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Filter df by a time_window dict:
      { "start": "DDD:HH:MM:SS.mmm", "end": "...", "time_column": "Elapsed Time (s)" }
    - Defaults to time_column='Elapsed Time (s)' if present.
    - Silently returns df if parsing fails or column missing.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if not isinstance(time_window, dict):
        return df

    time_col = time_window.get("time_column")
    if not time_col:
        time_col = "Elapsed Time (s)" if "Elapsed Time (s)" in df.columns else None
    if not time_col or time_col not in df.columns:
        return df

    start_s = _parse_flight_time_index(str(time_window.get("start", ""))) if time_window.get("start") else None
    end_s = _parse_flight_time_index(str(time_window.get("end", ""))) if time_window.get("end") else None
    try:
        x = pd.to_numeric(df[time_col], errors="coerce")
        mask = pd.Series(True, index=df.index)
        if start_s is not None:
            mask &= x >= start_s
        if end_s is not None:
            mask &= x <= end_s
        sub = df.loc[mask]
        return sub
    except Exception:
        return df


class ToolEnabledLLM:
    """
    Wrapper around OpenAI Chat with tool calling to:
      - inspect DataFrame schema
      - compute basic stats on columns (optionally within a time window)
      - generate well-structured tables
      - propose actionable calculation suggestions
      - query uploaded data embeddings (if available)
      - NEW: anomaly/trend/correlation analysis via FlightDataStatistics
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_rounds: int = 4,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1")
        self.max_rounds = max_rounds
        self.client = OpenAI(api_key=self.api_key)
        # Store tables the model asks to generate via tool calls
        self.generated_tables: List[Tuple[str, pd.DataFrame]] = []
        # Store calculation suggestions (textual)
        self.suggestions: List[Dict[str, Any]] = []
        # Internal hinting (from QueryAnalyzer)
        self._detected_time_window: Optional[Dict[str, str]] = None
        # NEW: store generated figures for UI rendering
        self.generated_figures: List[Tuple[str, Any]] = []

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
            # Ensure this schema exists (used by Report Assistant hybrid data retrieval)
            {
                "type": "function",
                "function": {
                    "name": "query_uploaded_data",
                    "description": "Semantic search over uploaded data embeddings to retrieve relevant rows/columns/summaries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Natural language query to search in uploaded data"},
                            "k": {"type": "integer", "description": "Top results to return", "default": 5},
                            "file_id": {"type": "string", "description": "Optional file ID to limit search"},
                            "embedding_type": {
                                "type": "string",
                                "enum": ["row", "column", "summary", "all"],
                                "description": "Which embeddings to search",
                                "default": "all"
                            },
                        },
                        "required": ["query"],
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
                            "description": "Compute basic statistics for specified numeric columns. Optionally restrict by a time window.",
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
                                    "time_window": {
                                        "type": "object",
                                        "description": "Optional time window to filter rows using Elapsed Time (s) or another numeric time column. Supports DDD:HH:MM:SS.mmm.",
                                        "properties": {
                                            "start": {"type": "string", "description": "Start time (e.g., 001:10:15:00.000)"},
                                            "end": {"type": "string", "description": "End time (e.g., 001:10:20:00.000)"},
                                            "time_column": {"type": "string", "description": "Time column name, defaults to 'Elapsed Time (s)' if present."}
                                        }
                                    }
                                },
                                "required": ["columns"],
                            },
                        },
                    },
                    # NEW: Anomaly detection tool (IQR/Z-score/Modified Z)
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_anomalies",
                            "description": "Detect outliers/anomalies in specified columns using IQR, Z-score, or Modified Z-score. Optionally restrict by a time window.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Columns to analyze for anomalies",
                                    },
                                    "method": {
                                        "type": "string",
                                        "enum": ["iqr", "zscore", "modified_zscore"],
                                        "default": "iqr",
                                        "description": "Outlier detection method"
                                    },
                                    "time_window": {
                                        "type": "object",
                                        "description": "Optional time window to filter rows. Supports DDD:HH:MM:SS.mmm.",
                                        "properties": {
                                            "start": {"type": "string"},
                                            "end": {"type": "string"},
                                            "time_column": {"type": "string"}
                                        }
                                    }
                                },
                                "required": ["columns"],
                            },
                        },
                    },
                    # NEW: Trend analysis tool
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_trends",
                            "description": "Perform linear trend analysis on columns over time. Returns slope, R², p-value, and direction. Optionally restrict by a time window.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Columns to analyze for trends",
                                    },
                                    "time_column": {
                                        "type": "string",
                                        "description": "Time column name, default 'Elapsed Time (s)'"
                                    },
                                    "time_window": {
                                        "type": "object",
                                        "description": "Optional time window to filter rows. Supports DDD:HH:MM:SS.mmm.",
                                        "properties": {
                                            "start": {"type": "string"},
                                            "end": {"type": "string"},
                                            "time_column": {"type": "string"}
                                        }
                                    }
                                },
                                "required": ["columns"],
                            },
                        },
                    },
                    # NEW: Correlation analysis tool
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_correlations",
                            "description": "Compute correlation matrix and strongest pairs for specified columns. Optionally restrict by a time window.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Columns to include in correlation analysis",
                                    },
                                    "method": {
                                        "type": "string",
                                        "enum": ["pearson", "spearman", "kendall"],
                                        "default": "pearson",
                                        "description": "Correlation method"
                                    },
                                    "time_window": {
                                        "type": "object",
                                        "description": "Optional time window to filter rows. Supports DDD:HH:MM:SS.mmm.",
                                        "properties": {
                                            "start": {"type": "string"},
                                            "end": {"type": "string"},
                                            "time_column": {"type": "string"}
                                        }
                                    }
                                },
                                "required": ["columns"],
                            },
                        },
                    },
                    # NEW: Plotting tools
                    {
                        "type": "function",
                        "function": {
                            "name": "plot_time_series",
                            "description": "Plot one or more columns against a time column as lines.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "time_column": {
                                        "type": "string",
                                        "description": "Time column name, default 'Elapsed Time (s)'"
                                    },
                                    "y_columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "One or more numeric columns to plot"
                                    },
                                    "title": {"type": "string"},
                                    "time_window": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "string"},
                                            "end": {"type": "string"},
                                            "time_column": {"type": "string"}
                                        }
                                    }
                                },
                                "required": ["y_columns"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "plot_scatter",
                            "description": "Scatter plot for two columns, optionally color by a category.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "string"},
                                    "y": {"type": "string"},
                                    "color_by": {"type": "string", "description": "Optional categorical column"},
                                    "title": {"type": "string"},
                                    "time_window": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "string"},
                                            "end": {"type": "string"},
                                            "time_column": {"type": "string"}
                                        }
                                    }
                                },
                                "required": ["x", "y"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "plot_histogram",
                            "description": "Histogram for a numeric column.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "bins": {"type": "integer", "default": 30},
                                    "title": {"type": "string"},
                                    "time_window": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "string"},
                                            "end": {"type": "string"},
                                            "time_column": {"type": "string"}
                                        }
                                    }
                                },
                                "required": ["column"]
                            }
                        }
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

                # Optional time window filter
                sub_df = df
                tw = arguments.get("time_window")
                if isinstance(tw, dict) or self._detected_time_window:
                    # Prefer explicit tool arg; fallback to detected hint
                    tw_eff = tw or self._detected_time_window
                    sub_df = _apply_time_window_filter(df, tw_eff)

                numeric_cols = [c for c in cols if c in sub_df.columns and pd.api.types.is_numeric_dtype(sub_df[c])]
                if not numeric_cols:
                    return _safe_json_dumps({"error": "No valid numeric columns in request"})

                result_rows: List[Dict[str, Any]] = []
                for c in numeric_cols:
                    series = pd.to_numeric(sub_df[c], errors="coerce")
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
                        file_id=arguments.get("file_id") or None,
                    )
                    # No UI table here – Report Assistant renders citations
                    return _safe_json_dumps({"results": results})
                except Exception as e:
                    return _safe_json_dumps({"error": f"query_uploaded_data failed: {e}"})

            # NEW: anomaly detection
            if name == "analyze_anomalies":
                if df is None:
                    return _safe_json_dumps({"error": "No dataset available"})
                cols = [c for c in (arguments.get("columns") or []) if c in df.columns]
                if not cols:
                    return _safe_json_dumps({"error": "No valid columns specified"})
                method = arguments.get("method", "iqr") or "iqr"
                sub_df = df
                tw = arguments.get("time_window")
                if isinstance(tw, dict) or self._detected_time_window:
                    sub_df = _apply_time_window_filter(df, tw or self._detected_time_window)

                stats = FlightDataStatistics()
                try:
                    res = stats.detect_outliers(sub_df, cols, method=method)
                except Exception as e:
                    return _safe_json_dumps({"error": f"detect_outliers failed: {e}"})

                # Build compact table
                rows = []
                for param, info in (res or {}).items():
                    bounds = info.get("bounds", {}) if isinstance(info, dict) else {}
                    rows.append([
                        param,
                        int(info.get("outlier_count", 0)),
                        float(info.get("outlier_percentage", 0.0)),
                        str(info.get("method", method)).upper(),
                        bounds.get("lower"),
                        bounds.get("upper")
                    ])
                df_tbl = pd.DataFrame(rows, columns=["Parameter", "Outliers", "Outlier %", "Method", "Lower", "Upper"])
                self.generated_tables.append(("Anomaly Detection", df_tbl))
                return _safe_json_dumps({"ok": True, "results": res})

            # NEW: trend analysis
            if name == "analyze_trends":
                if df is None:
                    return _safe_json_dumps({"error": "No dataset available"})
                cols = [c for c in (arguments.get("columns") or []) if c in df.columns]
                if not cols:
                    return _safe_json_dumps({"error": "No valid columns specified"})
                time_col = arguments.get("time_column") or ("Elapsed Time (s)" if "Elapsed Time (s)" in df.columns else None)
                if not time_col or time_col not in df.columns:
                    return _safe_json_dumps({"error": "Time column not found. Provide 'time_column' or include 'Elapsed Time (s)'."})

                sub_df = df
                tw = arguments.get("time_window")
                if isinstance(tw, dict) or self._detected_time_window:
                    sub_df = _apply_time_window_filter(df, tw or self._detected_time_window)

                stats = FlightDataStatistics()
                try:
                    res = stats.perform_trend_analysis(sub_df, time_col, cols)
                except Exception as e:
                    return _safe_json_dumps({"error": f"perform_trend_analysis failed: {e}"})

                # Compact table
                rows = []
                for param, info in (res or {}).items():
                    lin = info.get("linear_trend", {}) if isinstance(info, dict) else {}
                    rows.append([
                        param,
                        lin.get("trend_direction", ""),
                        float(lin.get("slope", 0.0)),
                        float(lin.get("r_squared", 0.0)),
                        float(lin.get("p_value", 1.0)),
                        "Yes" if float(lin.get("p_value", 1.0)) < 0.05 else "No",
                        int(info.get("data_points", 0))
                    ])
                df_tbl = pd.DataFrame(rows, columns=["Parameter", "Direction", "Slope", "R²", "p-value", "Significant", "Points"])
                self.generated_tables.append(("Trend Analysis", df_tbl))
                return _safe_json_dumps({"ok": True, "results": res})

            # NEW: correlation analysis
            if name == "analyze_correlations":
                if df is None:
                    return _safe_json_dumps({"error": "No dataset available"})
                cols = [c for c in (arguments.get("columns") or []) if c in df.columns]
                if not cols or len(cols) < 2:
                    return _safe_json_dumps({"error": "Provide at least two valid columns"})
                method = arguments.get("method", "pearson") or "pearson"
                sub_df = df
                tw = arguments.get("time_window")
                if isinstance(tw, dict) or self._detected_time_window:
                    sub_df = _apply_time_window_filter(df, tw or self._detected_time_window)

                stats = FlightDataStatistics()
                try:
                    res = stats.compute_correlation_analysis(sub_df, cols, method=method)
                except Exception as e:
                    return _safe_json_dumps({"error": f"compute_correlation_analysis failed: {e}"})

                # Top pairs table (if provided)
                pairs = res.get("strongest_correlations", []) if isinstance(res, dict) else []
                pair_rows = []
                for it in pairs[:10]:
                    pair_rows.append([it.get("param1"), it.get("param2"), float(it.get("correlation", 0.0))])
                if pair_rows:
                    df_pairs = pd.DataFrame(pair_rows, columns=["Param 1", "Param 2", "Correlation"])
                    self.generated_tables.append(("Top Correlations", df_pairs))

                return _safe_json_dumps({"ok": True, "results": res})
            
            # NEW: plotting handlers
            if name == "plot_time_series":
                if df is None or df.empty:
                    return _safe_json_dumps({"error": "No dataset available"})
                y_cols = [c for c in (arguments.get("y_columns") or []) if c in df.columns]
                if not y_cols:
                    return _safe_json_dumps({"error": "No valid y_columns"})
                time_col = arguments.get("time_column") or ("Elapsed Time (s)" if "Elapsed Time (s)" in df.columns else None)
                if not time_col or time_col not in df.columns:
                    return _safe_json_dumps({"error": "Time column not found. Provide 'time_column' or include 'Elapsed Time (s)'."})
                sub_df = df
                tw = arguments.get("time_window")
                if isinstance(tw, dict) or self._detected_time_window:
                    sub_df = _apply_time_window_filter(df, tw or self._detected_time_window)

                plt.figure(figsize=(8, 4.5))
                for c in y_cols:
                    try:
                        plt.plot(pd.to_numeric(sub_df[time_col], errors="coerce"),
                                 pd.to_numeric(sub_df[c], errors="coerce"),
                                 label=c, linewidth=1.2)
                    except Exception:
                        continue
                plt.xlabel(time_col)
                plt.ylabel(", ".join(y_cols))
                plt.title(arguments.get("title") or f"Time Series: {', '.join(y_cols)} vs {time_col}")
                plt.legend(loc="best", fontsize=8)
                plt.grid(True, alpha=0.3)
                fig = plt.gcf()
                self.generated_figures.append((arguments.get("title") or "Time Series", fig))
                return _safe_json_dumps({"ok": True})
            
            if name == "plot_scatter":
                if df is None or df.empty:
                    return _safe_json_dumps({"error": "No dataset available"})
                x = arguments.get("x")
                y = arguments.get("y")
                color_by = arguments.get("color_by")
                if x not in df.columns or y not in df.columns:
                    return _safe_json_dumps({"error": "x or y not found in dataset"})
                sub_df = df
                tw = arguments.get("time_window")
                if isinstance(tw, dict) or self._detected_time_window:
                    sub_df = _apply_time_window_filter(df, tw or self._detected_time_window)

                plt.figure(figsize=(6.5, 5))
                if color_by and color_by in sub_df.columns:
                    for val, chunk in sub_df.groupby(color_by):
                        plt.scatter(pd.to_numeric(chunk[x], errors="coerce"),
                                    pd.to_numeric(chunk[y], errors="coerce"),
                                    s=12, alpha=0.7, label=str(val))
                    plt.legend(loc="best", fontsize=8, title=color_by)
                else:
                    plt.scatter(pd.to_numeric(sub_df[x], errors="coerce"),
                                pd.to_numeric(sub_df[y], errors="coerce"),
                                s=12, alpha=0.7)
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(arguments.get("title") or f"Scatter: {y} vs {x}")
                plt.grid(True, alpha=0.3)
                fig = plt.gcf()
                self.generated_figures.append((arguments.get("title") or "Scatter", fig))
                return _safe_json_dumps({"ok": True})
            
            if name == "plot_histogram":
                if df is None or df.empty:
                    return _safe_json_dumps({"error": "No dataset available"})
                col = arguments.get("column")
                if col not in df.columns:
                    return _safe_json_dumps({"error": "Column not found"})
                bins = int(arguments.get("bins", 30))
                sub_df = df
                tw = arguments.get("time_window")
                if isinstance(tw, dict) or self._detected_time_window:
                    sub_df = _apply_time_window_filter(df, tw or self._detected_time_window)

                plt.figure(figsize=(6.5, 4.5))
                plt.hist(pd.to_numeric(sub_df[col], errors="coerce").dropna(), bins=bins, alpha=0.85, color="#1f77b4")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.title(arguments.get("title") or f"Histogram: {col}")
                plt.grid(True, alpha=0.3)
                fig = plt.gcf()
                self.generated_figures.append((arguments.get("title") or "Histogram", fig))
                return _safe_json_dumps({"ok": True})

            return _safe_json_dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return _safe_json_dumps({"error": f"Tool '{name}' failed: {e}"})


    def ask(
        self,
        prompt: str,
        system_prompt: str,
        df: Optional[pd.DataFrame] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.6,
        max_tokens: int = 4000,
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
            "suggestions": List[Dict],
            "history": List[Dict[str, Any]]  # NEW: The updated conversation history
          }
        """
        self.generated_tables.clear()
        self.suggestions.clear()
        self.generated_figures.clear()
        self._detected_time_window = None

        # --- CONVERSATION HISTORY MANAGEMENT ---
        messages = chat_history.copy() if chat_history else []
        
        # If history is empty, create the system prompt
        if not messages:
            schema_hint = ""
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                sample_cols = df.columns.tolist()[:50]  # cap to be safe
                schema_hint = (
                    "Dataset schema (first 50 columns shown):\n"
                    + ", ".join(sample_cols)
                    + "\nIf you need to compute stats, detect anomalies, analyze trends, or produce tables, use the available tools."
                )

            detail_instruction = {
                "brief": "Be concise. Prioritize high-level insights. Use tables only if critical.",
                "standard": "Be comprehensive but focused. Use compact tables where they clarify results.",
                "deep": "Provide an-depth analysis with step-by-step reasoning, multiple perspectives, and supporting tables where useful.",
            }.get(detail_level, "Be comprehensive but focused.")

            system = (
                f"{system_prompt}\n"
                f"- {detail_instruction}\n"
                "- When useful, structure the answer with markdown headings: '### Summary', '### Details', '### Recommendations'.\n"
                # >>> NEW: Added instruction for LaTeX formulas
                "- For mathematical formulas, use LaTeX syntax enclosed in `$` or `$$` delimiters (e.g., `$\\Delta V / \\Delta t$`).\n"
                "- Only cite sources when explicitly provided in the prompt/context.\n"
            )

            if enable_tools:
                system += "- Use tools for data inspection, statistics, anomaly detection, trend, correlation, and emitting tabular outputs; do not fabricate numeric results.\n"
            else:
                system += "- Do not invent specific numeric values; describe the method to compute them if needed.\n"

            if schema_hint:
                system += "\n" + schema_hint
            
            if extra_context:
                system += "\n" + extra_context
            
            messages.append({"role": "system", "content": system})
        
        # --- END SYSTEM PROMPT SETUP ---

        # NEW: Use a simple QueryAnalyzer to detect time window hints from NL prompt
        if prompt and df is not None and not df.empty:
            try:
                qa = QueryAnalyzer()
                tw = qa.detect_time_window(prompt)
                # If analyzer found a window, record and hint to the model
                if tw and (tw.get("start") or tw.get("end")):
                    self._detected_time_window = tw
            except Exception:
                pass

        # Hint the model about any detected time window so it can pass it in tool calls
        if self._detected_time_window:
            det = self._detected_time_window
            # Add a temporary system message with the hint
            time_hint = (
                "\nDetected time window from the user's query: "
                f"{det.get('start','?')} → {det.get('end','?')} "
                f"(time_column='{det.get('time_column','Elapsed Time (s)')}'). "
                "When calling tools, include this as the 'time_window' parameter."
            )
            # Find and update system prompt, or add a new one if not found
            system_found = False
            for msg in messages:
                if msg['role'] == 'system':
                    msg['content'] += time_hint
                    system_found = True
                    break
            if not system_found:
                 messages.insert(0, {"role": "system", "content": time_hint})


        # Add the current user prompt to the history
        messages.append({"role": "user", "content": prompt})

        tools = self._build_tool_schemas(df) if enable_tools else None

        # Tool loop
        for _ in range(self.max_rounds):
            # Prune history if it's getting too long (simple strategy)
            # Keep system prompt, and last N exchanges
            if len(messages) > 15:
                messages = [messages[0]] + messages[-14:]

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if enable_tools else None,
                tool_choice="auto" if enable_tools else None,
            )
            choice = resp.choices[0]
            response_message = choice.message

            # If tool calls exist, handle them
            if enable_tools and response_message.tool_calls:
                # Append assistant message with tool_calls to history
                messages.append(response_message)
                
                # Execute each tool call and append results
                for tc in response_message.tool_calls:
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
            text = response_message.content or ""
            sections = _sectionize_markdown(text)
            
            # Append final assistant response to history
            messages.append(response_message)
            
            return {
                "text": text,
                "sections": sections,
                "tables": list(self.generated_tables),
                "suggestions": list(self.suggestions),
                "figures": list(self.generated_figures),  # NEW
                "history": messages, # Return the full history
            }

        # If loop exits without a normal return
        final_text = "The assistant reached the maximum tool-calling steps without concluding."
        messages.append({"role": "assistant", "content": final_text})
        return {
            "text": final_text,
            "sections": {},
            "tables": list(self.generated_tables),
            "suggestions": list(self.suggestions),
            "figures": list(self.generated_figures),  # NEW
            "history": messages,
        }