"""
Agent orchestration pipeline.
Provides a single AnalystPipeline class that chains all agents together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from app.agents.cleaning_agent import CleaningAgent, CleaningConfig, CleaningReport
from app.agents.data_understanding import DataUnderstandingAgent, DatasetProfile
from app.agents.eda_agent import EDAAgent, EDAResults
from app.agents.insight_generator import InsightGenerationAgent, InsightReport
from app.agents.query_agent import QueryAgent, QueryResult
from app.agents.report_generator import ReportGenerator
from app.utils.data_loader import load_dataset


@dataclass
class PipelineResult:
    df_raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_clean: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile: DatasetProfile = field(default_factory=DatasetProfile)
    cleaning_report: CleaningReport = field(default_factory=CleaningReport)
    eda_results: EDAResults = field(default_factory=EDAResults)
    insight_report: InsightReport = field(default_factory=InsightReport)
    html_report_path: str = ""
    pdf_report_path: str = ""
    query_agent: QueryAgent | None = None


class AnalystPipeline:
    """
    Orchestrates all 7 modules in sequence:
    load → understand → clean → EDA → insights → report
    """

    def __init__(
        self,
        *,
        use_llm: bool = True,
        cleaning_config: CleaningConfig | None = None,
        save_charts: bool = True,
        generate_pdf: bool = False,
    ):
        self.use_llm = use_llm
        self.cleaning_config = cleaning_config or CleaningConfig()
        self.save_charts = save_charts
        self.generate_pdf = generate_pdf

    # ── Public ────────────────────────────────────────────────────────────────

    def run_from_file(
        self,
        source,
        filename: str = "",
        business_context: str = "",
        title: str = "Data Analysis Report",
    ) -> PipelineResult:
        df = load_dataset(source, filename)
        return self.run(df, business_context=business_context, title=title)

    def run(
        self,
        df: pd.DataFrame,
        business_context: str = "",
        title: str = "Data Analysis Report",
    ) -> PipelineResult:
        result = PipelineResult(df_raw=df)

        logger.info("=== AnalystPipeline starting ===")

        # Module 1 — Understand
        understanding = DataUnderstandingAgent(use_llm=self.use_llm)
        result.profile = understanding.run(df, business_context=business_context)

        # Module 2 — Clean
        cleaner = CleaningAgent(config=self.cleaning_config)
        result.df_clean, result.cleaning_report = cleaner.run(df, result.profile)

        # Module 3 — EDA
        eda = EDAAgent(save_charts=self.save_charts)
        result.eda_results = eda.run(result.df_clean, result.profile)

        # Module 4 — Insights
        insight_gen = InsightGenerationAgent(use_llm=self.use_llm)
        result.insight_report = insight_gen.run(
            result.df_clean,
            result.profile,
            result.eda_results,
            result.cleaning_report,
            business_context=business_context,
        )

        # Module 7 — Report
        reporter = ReportGenerator()
        result.html_report_path = reporter.generate_html(
            profile=result.profile,
            cleaning_report=result.cleaning_report,
            eda_results=result.eda_results,
            insight_report=result.insight_report,
            business_context=business_context,
            title=title,
        )
        if self.generate_pdf:
            result.pdf_report_path = reporter.generate_pdf(result.html_report_path) or ""

        # Module 5 — Query agent (ready, but not run yet)
        result.query_agent = QueryAgent(
            df=result.df_clean, profile=result.profile
        )

        logger.info("=== AnalystPipeline complete ===")
        return result
