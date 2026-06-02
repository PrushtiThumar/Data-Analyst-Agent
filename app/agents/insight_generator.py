"""
Module 4 — Insight Generation Agent
Converts raw statistics into human-readable, actionable business insights
using the LLM with rich statistical context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from app.agents.cleaning_agent import CleaningReport
from app.agents.data_understanding import DatasetProfile
from app.agents.eda_agent import EDAResults
from app.config import MAX_INSIGHTS
from app.utils.llm_client import chat_completion


@dataclass
class InsightReport:
    insights: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    key_drivers: list[str] = field(default_factory=list)
    data_quality_notes: list[str] = field(default_factory=list)


class InsightGenerationAgent:
    """
    Generates human-readable insights from EDA results and the dataset profile.
    Falls back to rule-based insights when LLM is unavailable.
    """

    def __init__(self, use_llm: bool = True, max_insights: int = MAX_INSIGHTS):
        self.use_llm = use_llm
        self.max_insights = max_insights

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        eda: EDAResults,
        cleaning_report: CleaningReport,
        business_context: str = "",
    ) -> InsightReport:
        logger.info("InsightGenerationAgent: generating insights...")

        report = InsightReport()

        # Rule-based baseline (always run — fast, no API)
        report.data_quality_notes = self._data_quality_notes(profile, cleaning_report)
        rule_insights = self._rule_based_insights(df, profile, eda)

        if self.use_llm:
            llm_insights = self._llm_insights(
                profile, eda, business_context, rule_insights
            )
            report.insights = llm_insights.get("insights", rule_insights)
            report.recommendations = llm_insights.get("recommendations", [])
            report.anomalies = llm_insights.get("anomalies", [])
            report.key_drivers = llm_insights.get("key_drivers", [])
        else:
            report.insights = rule_insights

        logger.info(
            f"Generated {len(report.insights)} insights, "
            f"{len(report.recommendations)} recommendations."
        )
        return report

    # ── Rule-based ────────────────────────────────────────────────────────────

    def _rule_based_insights(
        self, df: pd.DataFrame, profile: DatasetProfile, eda: EDAResults
    ) -> list[str]:
        insights: list[str] = []

        # 1. Correlation insights
        if eda.correlation_matrix:
            corr_df = pd.DataFrame(eda.correlation_matrix)
            pairs = self._top_correlations(corr_df, n=3)
            for col_a, col_b, r in pairs:
                direction = "positively" if r > 0 else "negatively"
                strength = "strongly" if abs(r) > 0.7 else "moderately"
                insights.append(
                    f"{col_a} and {col_b} are {strength} {direction} correlated "
                    f"(r = {r:.2f}), suggesting a meaningful relationship."
                )

        # 2. Skewness
        for col, skew in eda.skewness.items():
            if abs(skew) > 1.0:
                direction = "right (positively)" if skew > 0 else "left (negatively)"
                insights.append(
                    f"{col} is {direction} skewed (skewness = {skew:.2f}), "
                    "indicating the presence of outliers or a non-normal distribution."
                )

        # 3. Categorical dominance
        for col, stats in eda.categorical_stats.items():
            vc = stats.get("top_values", {})
            if not vc:
                continue
            total = sum(vc.values())
            top_val = stats.get("top_value")
            top_freq = stats.get("top_freq", 0)
            if total > 0 and top_freq / total > 0.5:
                pct = round(top_freq / total * 100, 1)
                insights.append(
                    f"'{top_val}' dominates the {col} column, representing "
                    f"{pct}% of all records — consider whether class imbalance "
                    "may affect model performance."
                )

        # 4. Numerical range flags
        for col, stats in eda.numerical_stats.items():
            cv = stats["std"] / abs(stats["mean"]) if stats["mean"] != 0 else 0
            if cv > 1.5:
                insights.append(
                    f"{col} has a very high coefficient of variation ({cv:.2f}), "
                    "indicating extreme variability across records."
                )

        # 5. Missing value highlights
        high_missing = {
            k: v for k, v in profile.missing_pct.items() if 10 < v <= 50
        }
        if high_missing:
            cols_str = ", ".join(
                f"{k} ({v:.1f}%)" for k, v in list(high_missing.items())[:3]
            )
            insights.append(
                f"Several columns have notable missing data: {cols_str}. "
                "Imputation choices may significantly influence results."
            )

        return insights[: self.max_insights]

    def _data_quality_notes(
        self, profile: DatasetProfile, cleaning_report: CleaningReport
    ) -> list[str]:
        notes: list[str] = []
        if profile.duplicate_rows:
            notes.append(
                f"{profile.duplicate_rows:,} duplicate rows detected "
                f"({profile.duplicate_rows / max(profile.rows, 1) * 100:.1f}% of data)."
            )
        if cleaning_report.dropped_columns:
            notes.append(
                f"Columns dropped due to >50% missing values: "
                f"{cleaning_report.dropped_columns}"
            )
        if cleaning_report.outliers_handled:
            total = sum(cleaning_report.outliers_handled.values())
            notes.append(
                f"{total:,} outliers capped/removed across "
                f"{len(cleaning_report.outliers_handled)} columns."
            )
        return notes

    @staticmethod
    def _top_correlations(
        corr: pd.DataFrame, n: int = 3
    ) -> list[tuple[str, str, float]]:
        pairs: list[tuple[str, str, float]] = []
        cols = corr.columns.tolist()
        for i, ca in enumerate(cols):
            for cb in cols[i + 1:]:
                r = corr.loc[ca, cb]
                if pd.notna(r) and abs(r) >= 0.4:
                    pairs.append((ca, cb, round(float(r), 3)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs[:n]

    # ── LLM insights ─────────────────────────────────────────────────────────

    def _llm_insights(
        self,
        profile: DatasetProfile,
        eda: EDAResults,
        business_context: str,
        rule_insights: list[str],
    ) -> dict[str, list[str]]:
        # Build a concise stats summary to keep the prompt manageable
        stats_summary = self._build_stats_summary(profile, eda)

        prompt = f"""
You are a senior data analyst providing insights for a business stakeholder.

Business context: {business_context or "General data analysis"}

Dataset overview:
- {profile.rows:,} rows, {profile.columns} columns
- Numerical columns: {profile.numerical}
- Categorical columns: {profile.categorical}

Statistical highlights:
{stats_summary}

Rule-based observations already found:
{json.dumps(rule_insights, indent=2)}

Generate a comprehensive analysis. Return a JSON object with these keys:
- "insights": list of 8-12 human-readable insight strings (go beyond the rule-based ones; interpret meaning, not just numbers)
- "recommendations": list of 3-5 actionable business recommendations
- "anomalies": list of notable anomalies or data quality flags (2-4 items)
- "key_drivers": list of 2-4 columns most likely to drive outcomes

Each insight should read as a natural business observation, not a statistic.
For example: "High-income customers aged 30-45 account for the majority of revenue,
suggesting this segment warrants priority retention efforts."

Return ONLY valid JSON.
""".strip()

        try:
            raw = chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2048,
            )
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"LLM insight generation failed: {exc}. Using rule-based.")
            return {"insights": rule_insights}

    def _build_stats_summary(
        self, profile: DatasetProfile, eda: EDAResults
    ) -> str:
        lines: list[str] = []

        # Top correlations
        if eda.correlation_matrix:
            corr_df = pd.DataFrame(eda.correlation_matrix)
            pairs = self._top_correlations(corr_df, n=5)
            if pairs:
                lines.append("Top correlations:")
                for a, b, r in pairs:
                    lines.append(f"  {a} ↔ {b}: {r}")

        # Skewness flags
        skewed = {k: v for k, v in eda.skewness.items() if abs(v) > 1}
        if skewed:
            lines.append(f"Skewed columns: {skewed}")

        # Categorical top values
        for col, s in list(eda.categorical_stats.items())[:4]:
            lines.append(
                f"{col} top values: "
                + str(list(s.get("top_values", {}).keys())[:5])
            )

        # Numerical summary for key cols
        for col, s in list(eda.numerical_stats.items())[:5]:
            lines.append(
                f"{col}: mean={s['mean']}, std={s['std']}, median={s['median']}"
            )

        return "\n".join(lines) if lines else "No detailed stats available."
