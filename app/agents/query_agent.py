"""
Module 5 — Natural Language Query Agent
Translates user questions into pandas operations and returns natural language answers.
Uses LLM to generate Python code, executes it safely, then formats the result.
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from app.agents.data_understanding import DatasetProfile
from app.utils.llm_client import chat_completion


@dataclass
class QueryResult:
    question: str
    answer: str
    data: Any = None        # The raw computed result (df, series, scalar)
    code: str = ""
    error: str = ""
    success: bool = True


class QueryAgent:
    """
    Answers natural language questions about a DataFrame.
    Pipeline: question → LLM generates pandas code → safe exec → LLM narrates result.
    """

    # Blocked patterns to prevent code injection
    _BLOCKED_PATTERNS = [
        r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bsubprocess\b",
        r"\bopen\s*\(", r"\beval\s*\(", r"\bexec\s*\(", r"__import__",
        r"\bshutil\b", r"\bpathlib\b", r"\bos\.path\b",
    ]

    def __init__(self, df: pd.DataFrame, profile: DatasetProfile):
        self.df = df
        self.profile = profile
        self._schema_hint = self._build_schema_hint()

    # ── Public ────────────────────────────────────────────────────────────────

    def ask(self, question: str) -> QueryResult:
        """
        Answer a natural language question about the dataset.

        Args:
            question: Free-text question from the user.

        Returns:
            QueryResult with answer, raw data, and the generated code.
        """
        logger.info(f"QueryAgent: '{question}'")

        code = self._generate_code(question)
        if not code:
            return QueryResult(
                question=question,
                answer="I could not generate code for this question. Please rephrase.",
                success=False,
            )

        # Safety check
        if self._is_unsafe(code):
            return QueryResult(
                question=question,
                answer="This query was blocked for safety reasons.",
                success=False,
                code=code,
            )

        result, error = self._execute(code)
        if error:
            return QueryResult(
                question=question,
                answer=f"I encountered an error while computing the answer: {error}",
                success=False,
                code=code,
                error=error,
            )

        answer = self._narrate(question, result, code)
        return QueryResult(
            question=question,
            answer=answer,
            data=result,
            code=code,
            success=True,
        )

    # ── Code generation ───────────────────────────────────────────────────────

    def _generate_code(self, question: str) -> str:
        prompt = f"""
You are a Python data analyst. You have a pandas DataFrame called `df`.

Schema:
{self._schema_hint}

Write a single Python expression or short code block (≤10 lines) that answers
this question: "{question}"

Rules:
- Assign the final result to a variable called `result`
- Do NOT import any modules
- Do NOT use os, sys, subprocess, open(), eval(), exec()
- Prefer pandas/numpy operations
- Keep it concise and correct

Return ONLY the Python code. No explanations. No markdown.
""".strip()

        try:
            code = chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            # Strip markdown fences if present
            code = re.sub(r"```python|```", "", code).strip()
            return code
        except Exception as exc:
            logger.warning(f"Code generation failed: {exc}")
            return ""

    # ── Execution ─────────────────────────────────────────────────────────────

    def _execute(self, code: str) -> tuple[Any, str]:
        local_ns: dict[str, Any] = {"df": self.df.copy(), "pd": pd, "np": np}
        try:
            exec(code, {"__builtins__": {}}, local_ns)  # noqa: S102
            result = local_ns.get("result", None)
            return result, ""
        except Exception:
            error = traceback.format_exc(limit=2)
            logger.warning(f"Code execution error:\n{error}")
            return None, error

    # ── Narration ─────────────────────────────────────────────────────────────

    def _narrate(self, question: str, result: Any, code: str) -> str:
        result_str = self._result_to_str(result)

        prompt = f"""
You are a data analyst giving a concise, friendly answer to a business user.

Question: "{question}"

The Python code produced this result:
{result_str}

Write a 1-3 sentence natural language answer that directly answers the question.
Be specific — include actual values from the result.
""".strip()

        try:
            return chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
        except Exception:
            return f"Result: {result_str}"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_schema_hint(self) -> str:
        lines = [f"DataFrame shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} cols"]
        lines.append("Columns (name: dtype):")
        for col in self.df.columns:
            sample = self.df[col].dropna().iloc[:3].tolist()
            lines.append(f"  {col}: {self.df[col].dtype} — e.g. {sample}")
        return "\n".join(lines)

    def _is_unsafe(self, code: str) -> bool:
        for pattern in self._BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Blocked unsafe pattern in generated code: {pattern}")
                return True
        return False

    @staticmethod
    def _result_to_str(result: Any, max_rows: int = 20) -> str:
        if result is None:
            return "No result"
        if isinstance(result, pd.DataFrame):
            return result.head(max_rows).to_string()
        if isinstance(result, pd.Series):
            return result.head(max_rows).to_string()
        return str(result)
