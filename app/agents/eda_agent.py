"""
Module 3 — EDA Agent
Computes descriptive statistics, distributions, correlations,
and generates a suite of Plotly visualizations saved to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
from scipy import stats as scipy_stats

from app.agents.data_understanding import DatasetProfile
from app.config import MAX_CHART_ROWS, VIZ_DIR


# ── Results dataclass ─────────────────────────────────────────────────────────

@dataclass
class EDAResults:
    numerical_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    categorical_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    correlation_matrix: dict[str, Any] = field(default_factory=dict)
    skewness: dict[str, float] = field(default_factory=dict)
    chart_paths: list[str] = field(default_factory=list)
    chart_figures: list[go.Figure] = field(default_factory=list)


# ── Agent ─────────────────────────────────────────────────────────────────────

class EDAAgent:
    """
    Performs exploratory data analysis and generates visualizations.
    """

    _PALETTE = px.colors.qualitative.Set2

    def __init__(self, save_charts: bool = True, output_dir: Path | None = None):
        self.save_charts = save_charts
        self.output_dir = output_dir or VIZ_DIR

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, profile: DatasetProfile) -> EDAResults:
        logger.info("EDAAgent: computing statistics and generating charts...")
        results = EDAResults()

        # Sample for large datasets to keep charts snappy
        df_plot = df.sample(n=min(MAX_CHART_ROWS, len(df)), random_state=42)

        results.numerical_stats = self._numerical_stats(df, profile)
        results.categorical_stats = self._categorical_stats(df, profile)
        results.correlation_matrix = self._correlation_matrix(df, profile)
        results.skewness = self._compute_skewness(df, profile)

        figs = self._generate_charts(df_plot, profile, results)
        results.chart_figures = figs

        if self.save_charts:
            results.chart_paths = self._save_charts(figs)

        logger.info(
            f"EDA complete. {len(figs)} charts generated."
        )
        return results

    # ── Statistics ────────────────────────────────────────────────────────────

    def _numerical_stats(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        num_cols = [c for c in profile.numerical if c in df.columns]
        if not num_cols:
            return stats
        desc = df[num_cols].describe().T
        for col in num_cols:
            row = desc.loc[col]
            stats[col] = {
                "count": int(row["count"]),
                "mean": round(float(row["mean"]), 4),
                "std": round(float(row["std"]), 4),
                "min": round(float(row["min"]), 4),
                "25%": round(float(row["25%"]), 4),
                "median": round(float(row["50%"]), 4),
                "75%": round(float(row["75%"]), 4),
                "max": round(float(row["max"]), 4),
                "skewness": round(float(df[col].skew()), 4),
                "kurtosis": round(float(df[col].kurtosis()), 4),
            }
        return stats

    def _categorical_stats(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        cat_cols = [c for c in profile.categorical if c in df.columns]
        for col in cat_cols:
            vc = df[col].value_counts(normalize=False)
            stats[col] = {
                "unique": int(df[col].nunique()),
                "top_values": vc.head(10).to_dict(),
                "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
            }
        return stats

    def _correlation_matrix(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> dict[str, Any]:
        num_cols = [c for c in profile.numerical if c in df.columns]
        if len(num_cols) < 2:
            return {}
        corr = df[num_cols].corr()
        return corr.to_dict()

    def _compute_skewness(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> dict[str, float]:
        num_cols = [c for c in profile.numerical if c in df.columns]
        return {col: round(float(df[col].skew()), 4) for col in num_cols}

    # ── Charts ────────────────────────────────────────────────────────────────

    def _generate_charts(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        results: EDAResults,
    ) -> list[go.Figure]:
        figs: list[go.Figure] = []
        num_cols = [c for c in profile.numerical if c in df.columns][:8]
        cat_cols = [c for c in profile.categorical if c in df.columns][:6]

        if num_cols:
            figs.append(self._histogram_grid(df, num_cols))
            figs.append(self._boxplot_grid(df, num_cols))

        if len(num_cols) >= 2:
            figs.append(self._correlation_heatmap(df, num_cols))
            # Scatter matrix for top 4 numerical cols
            figs.append(self._scatter_matrix(df, num_cols[:4]))

        if cat_cols:
            figs.append(self._bar_chart_grid(df, cat_cols))
            if cat_cols:
                figs.append(self._pie_chart(df, cat_cols[0]))

        if num_cols and cat_cols:
            figs.append(self._boxplot_by_category(df, num_cols[0], cat_cols[0]))

        figs.append(self._missing_values_chart(profile))

        return [f for f in figs if f is not None]

    def _histogram_grid(
        self, df: pd.DataFrame, cols: list[str]
    ) -> go.Figure:
        n = len(cols)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols)
        for i, col in enumerate(cols):
            r, c = divmod(i, ncols)
            data = df[col].dropna()
            fig.add_trace(
                go.Histogram(x=data, name=col, marker_color=self._PALETTE[i % len(self._PALETTE)]),
                row=r + 1,
                col=c + 1,
            )
        fig.update_layout(
            title="Distribution of Numerical Features",
            showlegend=False,
            height=300 * nrows,
            template="plotly_white",
        )
        return fig

    def _boxplot_grid(self, df: pd.DataFrame, cols: list[str]) -> go.Figure:
        fig = go.Figure()
        for i, col in enumerate(cols):
            fig.add_trace(
                go.Box(y=df[col].dropna(), name=col,
                       marker_color=self._PALETTE[i % len(self._PALETTE)])
            )
        fig.update_layout(
            title="Boxplots of Numerical Features",
            showlegend=False,
            template="plotly_white",
            height=450,
        )
        return fig

    def _correlation_heatmap(self, df: pd.DataFrame, cols: list[str]) -> go.Figure:
        corr = df[cols].corr().round(2)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.values.round(2),
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title="Correlation Heatmap",
            template="plotly_white",
            height=500,
            xaxis=dict(tickangle=45),
        )
        return fig

    def _scatter_matrix(self, df: pd.DataFrame, cols: list[str]) -> go.Figure:
        fig = px.scatter_matrix(
            df[cols].dropna(),
            dimensions=cols,
            title="Scatter Matrix",
            template="plotly_white",
            height=600,
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
        return fig

    def _bar_chart_grid(self, df: pd.DataFrame, cols: list[str]) -> go.Figure:
        n = len(cols)
        ncols = min(n, 2)
        nrows = (n + 1) // 2
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols)
        for i, col in enumerate(cols):
            r, c = divmod(i, ncols)
            vc = df[col].value_counts().head(10)
            fig.add_trace(
                go.Bar(
                    x=vc.index.astype(str).tolist(),
                    y=vc.values.tolist(),
                    name=col,
                    marker_color=self._PALETTE[i % len(self._PALETTE)],
                ),
                row=r + 1,
                col=c + 1,
            )
        fig.update_layout(
            title="Categorical Feature Distributions",
            showlegend=False,
            height=350 * nrows,
            template="plotly_white",
        )
        return fig

    def _pie_chart(self, df: pd.DataFrame, col: str) -> go.Figure:
        vc = df[col].value_counts().head(8)
        fig = go.Figure(
            data=go.Pie(
                labels=vc.index.astype(str).tolist(),
                values=vc.values.tolist(),
                hole=0.3,
            )
        )
        fig.update_layout(
            title=f"Distribution of {col}",
            template="plotly_white",
            height=400,
        )
        return fig

    def _boxplot_by_category(
        self, df: pd.DataFrame, num_col: str, cat_col: str
    ) -> go.Figure:
        fig = px.box(
            df,
            x=cat_col,
            y=num_col,
            color=cat_col,
            title=f"{num_col} by {cat_col}",
            template="plotly_white",
            height=400,
        )
        fig.update_layout(showlegend=False, xaxis=dict(tickangle=30))
        return fig

    def _missing_values_chart(self, profile: DatasetProfile) -> go.Figure | None:
        missing = {
            k: v for k, v in profile.missing_pct.items() if v > 0
        }
        if not missing:
            return None
        sorted_items = sorted(missing.items(), key=lambda x: x[1], reverse=True)
        cols, pcts = zip(*sorted_items)
        fig = go.Figure(
            go.Bar(
                x=list(pcts),
                y=list(cols),
                orientation="h",
                marker_color="#EF553B",
            )
        )
        fig.update_layout(
            title="Missing Value Percentages",
            xaxis_title="% Missing",
            template="plotly_white",
            height=max(250, 30 * len(cols)),
        )
        return fig

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_charts(self, figs: list[go.Figure]) -> list[str]:
        paths: list[str] = []
        names = [
            "01_histograms", "02_boxplots", "03_correlation_heatmap",
            "04_scatter_matrix", "05_bar_charts", "06_pie_chart",
            "07_boxplot_by_category", "08_missing_values",
        ]
        for i, fig in enumerate(figs):
            name = names[i] if i < len(names) else f"chart_{i:02d}"
            path = self.output_dir / f"{name}.html"
            fig.write_html(str(path))
            paths.append(str(path))
            logger.debug(f"Saved chart: {path}")
        return paths
