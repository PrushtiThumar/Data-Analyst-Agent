"""
Module 1 — Dataset Understanding Agent
Automatically profiles a DataFrame: types, missing values, cardinality,
potential targets, and a plain-English summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from app.utils.llm_client import chat_completion


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DatasetProfile:
    rows: int = 0
    columns: int = 0
    numerical: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    missing: dict[str, int] = field(default_factory=dict)
    missing_pct: dict[str, float] = field(default_factory=dict)
    unique_counts: dict[str, int] = field(default_factory=dict)
    dtypes: dict[str, str] = field(default_factory=dict)
    potential_targets: list[str] = field(default_factory=list)
    memory_mb: float = 0.0
    duplicate_rows: int = 0
    summary_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DataUnderstandingAgent:
    """
    Profiles a raw DataFrame and optionally enriches the profile with an
    LLM-generated natural-language summary.
    """

    # Columns whose names hint at being target/label columns
    _TARGET_HINTS = {
        "churn", "target", "label", "outcome", "fraud", "default",
        "converted", "clicked", "purchased", "survived", "cancelled",
        "churned", "attrition", "class", "y",
    }

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, business_context: str = "") -> DatasetProfile:
        """
        Full pipeline: structural analysis → target detection → LLM summary.

        Args:
            df:               Raw DataFrame.
            business_context: Optional free-text context from the user.

        Returns:
            Populated DatasetProfile.
        """
        logger.info("DataUnderstandingAgent: starting profile...")

        profile = DatasetProfile()
        profile.rows, profile.columns = df.shape
        profile.memory_mb = round(df.memory_usage(deep=True).sum() / 1_048_576, 2)
        profile.duplicate_rows = int(df.duplicated().sum())

        # Column-level stats
        for col in df.columns:
            dtype = df[col].dtype
            profile.dtypes[col] = str(dtype)
            profile.missing[col] = int(df[col].isna().sum())
            profile.missing_pct[col] = round(
                profile.missing[col] / max(profile.rows, 1) * 100, 2
            )
            profile.unique_counts[col] = int(df[col].nunique())

        # Classify columns by kind
        profile.numerical = list(df.select_dtypes(include="number").columns)
        profile.categorical = self._detect_categorical(df)
        profile.datetime = self._detect_datetime(df)
        profile.boolean = list(df.select_dtypes(include="bool").columns)

        # Potential target variables
        profile.potential_targets = self._detect_targets(df)

        # LLM summary
        if self.use_llm:
            profile.summary_text = self._generate_summary(profile, business_context)

        logger.info(
            f"Profile complete — {profile.rows} rows × {profile.columns} cols, "
            f"{len(profile.numerical)} numerical, {len(profile.categorical)} categorical"
        )
        return profile

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_categorical(self, df: pd.DataFrame) -> list[str]:
        """Return columns that are object/category dtype OR low-cardinality int cols."""
        cats = list(df.select_dtypes(include=["object", "category"]).columns)
        # Low-cardinality integer columns (≤ 20 unique values, not boolean)
        for col in df.select_dtypes(include="number").columns:
            if df[col].nunique() <= 20 and col not in self._detect_boolean_cols(df):
                if col not in cats:
                    cats.append(col)
        return cats

    def _detect_datetime(self, df: pd.DataFrame) -> list[str]:
        dt_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        # Also try to detect string columns that look like dates
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(20)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                dt_cols.append(col)
            except Exception:
                pass
        return list(dict.fromkeys(dt_cols))  # preserve order, deduplicate

    def _detect_boolean_cols(self, df: pd.DataFrame) -> list[str]:
        bools = list(df.select_dtypes(include="bool").columns)
        for col in df.select_dtypes(include="number").columns:
            uniq = set(df[col].dropna().unique())
            if uniq <= {0, 1}:
                bools.append(col)
        return bools

    def _detect_targets(self, df: pd.DataFrame) -> list[str]:
        targets = []
        for col in df.columns:
            if col.lower() in self._TARGET_HINTS:
                targets.append(col)
        # Binary columns are also candidate targets
        for col in df.columns:
            if col not in targets and df[col].nunique() == 2:
                targets.append(col)
        return targets[:5]  # cap at 5

    def _generate_summary(
        self, profile: DatasetProfile, business_context: str
    ) -> str:
        prompt = f"""
You are a senior data analyst. Given the following dataset profile, write a concise
2-3 paragraph natural-language summary suitable for a business stakeholder.
Mention: size, key column types, data quality issues, and what kind of analysis
this dataset is suited for.

Business context: {business_context or "Not provided"}

Profile:
{json.dumps(profile.to_dict(), indent=2)}

Write the summary now:
""".strip()

        try:
            return chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
        except Exception as exc:
            logger.warning(f"LLM summary skipped: {exc}")
            return (
                f"Dataset contains {profile.rows:,} rows and {profile.columns} columns. "
                f"Numerical columns: {len(profile.numerical)}. "
                f"Categorical columns: {len(profile.categorical)}. "
                f"Duplicate rows: {profile.duplicate_rows}."
            )
