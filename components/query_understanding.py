"""
Simple Query Analyzer for NL prompt hints.

Current capabilities:
- Detect a DDD:HH:MM:SS(.mmm) → DDD:HH:MM:SS(.mmm) time window in the user's query.
- Optionally infer the time column (defaults to 'Elapsed Time (s)').

Designed to provide lightweight hints to ToolEnabledLLM so tools can be called
with 'time_window' automatically by the model.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# DDD:HH:MM:SS(.mmm) pattern
_TIME_PATTERN = r"(?P<t>\b\d{1,3}:\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?\b)"

# Common phrasing to anchor ranges
_RANGE_PATTERNS = [
    rf"(?:from|between)\s+{_TIME_PATTERN}\s+(?:to|and)\s+{_TIME_PATTERN}",
    rf"{_TIME_PATTERN}\s*[-–]\s*{_TIME_PATTERN}",  # 001:.. - 001:..
]


class QueryAnalyzer:
    def __init__(self, default_time_column: str = "Elapsed Time (s)"):
        self.default_time_column = default_time_column

    def detect_time_window(self, text: str) -> Optional[Dict[str, str]]:
        """
        Detect a time window in the form "from 001:... to 001:..." or "001:... - 001:..."
        Returns:
          { "start": str, "end": str, "time_column": str }
        or None if not found.
        """
        if not isinstance(text, str) or not text.strip():
            return None

        # Try anchored patterns first
        for pat in _RANGE_PATTERNS:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                # Extract all time-like matches in order
                times = re.findall(_TIME_PATTERN, m.group(0))
                if len(times) >= 2:
                    return {
                        "start": times[0],
                        "end": times[1],
                        "time_column": self.default_time_column,
                    }

        # Fallback: any two time tokens in the text interpreted as a window
        all_times = re.findall(_TIME_PATTERN, text)
        if len(all_times) >= 2:
            return {
                "start": all_times[0],
                "end": all_times[1],
                "time_column": self.default_time_column,
            }

        return None