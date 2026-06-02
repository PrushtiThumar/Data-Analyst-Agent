"""
Unit tests for the AI Data Analyst Agent.
Run with: pytest app/tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.agents.cleaning_agent import CleaningAgent, CleaningConfig
from app.agents.data_understanding import DataUnderstandingAgent
from app.agents.eda_agent import EDAAgent
from app.agents.insight_generator import InsightGenerationAgent
from app.agents.query_agent import QueryAgent
from app.utils.data_loader import load_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small realistic dataset for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(18, 70, n).astype(float),
        "income": np.random.normal(50_000, 15_000, n),
        "spend": np.random.normal(2_000, 800, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "gender": np.random.choice(["M", "F"], n),
        "churn": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="3D"),
        "score": np.where(np.random.rand(n) < 0.1, np.nan, np.random.rand(n) * 100),
    })


@pytest.fixture
def profile(sample_df):
    agent = DataUnderstandingAgent(use_llm=False)
    return agent.run(sample_df)


# ── DataUnderstandingAgent ────────────────────────────────────────────────────

class TestDataUnderstandingAgent:
    def test_basic_profile(self, sample_df):
        agent = DataUnderstandingAgent(use_llm=False)
        p = agent.run(sample_df)
        assert p.rows == 200
        assert p.columns == 8
        assert "age" in p.numerical
        assert "income" in p.numerical
        assert "region" in p.categorical
        assert "gender" in p.categorical

    def test_detects_target(self, sample_df):
        agent = DataUnderstandingAgent(use_llm=False)
        p = agent.run(sample_df)
        assert "churn" in p.potential_targets

    def test_missing_detection(self, sample_df):
        agent = DataUnderstandingAgent(use_llm=False)
        p = agent.run(sample_df)
        assert p.missing["score"] > 0
        assert p.missing_pct["score"] > 0

    def test_datetime_detection(self, sample_df):
        agent = DataUnderstandingAgent(use_llm=False)
        p = agent.run(sample_df)
        assert "signup_date" in p.datetime

    def test_no_duplicates(self, sample_df):
        agent = DataUnderstandingAgent(use_llm=False)
        p = agent.run(sample_df)
        assert p.duplicate_rows == 0

    def test_with_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df.iloc[:10]])
        agent = DataUnderstandingAgent(use_llm=False)
        p = agent.run(df_dup)
        assert p.duplicate_rows == 10


# ── CleaningAgent ─────────────────────────────────────────────────────────────

class TestCleaningAgent:
    def test_removes_duplicates(self, sample_df, profile):
        df_dup = pd.concat([sample_df, sample_df.iloc[:5]])
        agent = CleaningAgent(CleaningConfig(remove_duplicates=True))
        clean, report = agent.run(df_dup, profile)
        assert report.duplicates_removed == 5
        assert len(clean) == len(sample_df)

    def test_imputes_missing(self, sample_df, profile):
        agent = CleaningAgent(CleaningConfig(numerical_impute="median"))
        clean, report = agent.run(sample_df, profile)
        assert clean["score"].isna().sum() == 0
        assert "score" in report.imputed_columns

    def test_caps_outliers(self, sample_df, profile):
        df_out = sample_df.copy()
        df_out.loc[0, "income"] = 9_999_999  # extreme outlier
        agent = CleaningAgent(CleaningConfig(outlier_method="iqr", cap_outliers=True))
        clean, report = agent.run(df_out, profile)
        assert clean["income"].max() < 9_999_999
        assert "income" in report.outliers_handled

    def test_no_data_loss_with_drop_threshold(self, sample_df, profile):
        agent = CleaningAgent(CleaningConfig(missing_threshold=0.5))
        clean, report = agent.run(sample_df, profile)
        # score has ~10% missing — should NOT be dropped
        assert "score" in clean.columns
        assert report.dropped_columns == []

    def test_date_feature_engineering(self, sample_df, profile):
        agent = CleaningAgent(CleaningConfig(parse_dates=True, create_date_parts=True))
        clean, report = agent.run(sample_df, profile)
        assert "signup_date_year" in clean.columns
        assert "signup_date_month" in clean.columns


# ── EDAAgent ──────────────────────────────────────────────────────────────────

class TestEDAAgent:
    def test_numerical_stats_computed(self, sample_df, profile):
        agent = EDAAgent(save_charts=False)
        results = agent.run(sample_df, profile)
        assert "age" in results.numerical_stats
        assert "mean" in results.numerical_stats["age"]
        assert "median" in results.numerical_stats["age"]

    def test_categorical_stats_computed(self, sample_df, profile):
        agent = EDAAgent(save_charts=False)
        results = agent.run(sample_df, profile)
        assert "region" in results.categorical_stats
        assert results.categorical_stats["region"]["unique"] == 4

    def test_correlation_matrix_computed(self, sample_df, profile):
        agent = EDAAgent(save_charts=False)
        results = agent.run(sample_df, profile)
        assert "age" in results.correlation_matrix

    def test_charts_generated(self, sample_df, profile):
        agent = EDAAgent(save_charts=False)
        results = agent.run(sample_df, profile)
        assert len(results.chart_figures) > 0

    def test_skewness_computed(self, sample_df, profile):
        agent = EDAAgent(save_charts=False)
        results = agent.run(sample_df, profile)
        assert "income" in results.skewness


# ── InsightGenerationAgent ────────────────────────────────────────────────────

class TestInsightGenerationAgent:
    def test_rule_based_insights_generated(self, sample_df, profile):
        from app.agents.cleaning_agent import CleaningReport
        from app.agents.eda_agent import EDAResults

        eda_agent = EDAAgent(save_charts=False)
        eda_results = eda_agent.run(sample_df, profile)

        agent = InsightGenerationAgent(use_llm=False)
        report = agent.run(
            sample_df, profile, eda_results, CleaningReport(), business_context=""
        )
        assert len(report.insights) > 0

    def test_data_quality_notes(self, sample_df, profile):
        from app.agents.cleaning_agent import CleaningReport
        from app.agents.eda_agent import EDAResults

        eda_agent = EDAAgent(save_charts=False)
        eda_results = eda_agent.run(sample_df, profile)

        cr = CleaningReport()
        cr.dropped_columns = ["old_col"]
        agent = InsightGenerationAgent(use_llm=False)
        report = agent.run(sample_df, profile, eda_results, cr)
        assert any("old_col" in n for n in report.data_quality_notes)


# ── QueryAgent ────────────────────────────────────────────────────────────────

class TestQueryAgent:
    def test_blocks_unsafe_code(self, sample_df, profile):
        agent = QueryAgent(sample_df, profile)
        unsafe = "import os; result = os.listdir('/')"
        assert agent._is_unsafe(unsafe)

    def test_allows_safe_code(self, sample_df, profile):
        agent = QueryAgent(sample_df, profile)
        safe = "result = df['age'].mean()"
        assert not agent._is_unsafe(safe)

    def test_execute_safe_code(self, sample_df, profile):
        agent = QueryAgent(sample_df, profile)
        result, error = agent._execute("result = df['age'].mean()")
        assert error == ""
        assert isinstance(result, float)

    def test_execute_bad_code_returns_error(self, sample_df, profile):
        agent = QueryAgent(sample_df, profile)
        _, error = agent._execute("result = df['nonexistent_col'].mean()")
        assert error != ""


# ── Data Loader ───────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_csv(self, tmp_path, sample_df):
        csv_path = tmp_path / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_dataset(csv_path)
        assert df.shape[0] == sample_df.shape[0]

    def test_load_excel(self, tmp_path, sample_df):
        xl_path = tmp_path / "test.xlsx"
        sample_df.to_excel(xl_path, index=False)
        df = load_dataset(xl_path)
        assert df.shape[0] == sample_df.shape[0]

    def test_load_json(self, tmp_path, sample_df):
        j_path = tmp_path / "test.json"
        sample_df.to_json(j_path, orient="records")
        df = load_dataset(j_path)
        assert df.shape[0] == sample_df.shape[0]

    def test_unsupported_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(tmp_path / "data.parquet")
