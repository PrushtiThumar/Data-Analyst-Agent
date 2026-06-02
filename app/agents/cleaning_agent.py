"""
Module 2 — Data Cleaning Agent
Handles missing values, duplicates, outliers, and basic feature engineering.
Returns a cleaned DataFrame plus a human-readable cleaning report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from app.agents.data_understanding import DatasetProfile


# ── Config ────────────────────────────────────────────────────────────────────

ImpStrategy = Literal["mean", "median", "mode", "drop"]
OutlierMethod = Literal["iqr", "zscore", "none"]


@dataclass
class CleaningConfig:
    # Missing value strategies
    numerical_impute: ImpStrategy = "median"
    categorical_impute: ImpStrategy = "mode"
    missing_threshold: float = 0.5   # Drop column if > 50 % missing

    # Outlier handling
    outlier_method: OutlierMethod = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    cap_outliers: bool = True         # Cap instead of drop

    # Deduplication
    remove_duplicates: bool = True

    # Feature engineering
    parse_dates: bool = True
    create_date_parts: bool = True    # year / month / dayofweek from datetime cols


@dataclass
class CleaningReport:
    original_shape: tuple[int, int] = (0, 0)
    final_shape: tuple[int, int] = (0, 0)
    dropped_columns: list[str] = field(default_factory=list)
    imputed_columns: dict[str, str] = field(default_factory=dict)
    duplicates_removed: int = 0
    outliers_handled: dict[str, int] = field(default_factory=dict)
    date_columns_parsed: list[str] = field(default_factory=list)
    new_columns: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Original: {self.original_shape[0]:,} rows × {self.original_shape[1]} cols",
            f"Final:    {self.final_shape[0]:,} rows × {self.final_shape[1]} cols",
        ]
        if self.duplicates_removed:
            lines.append(f"Removed {self.duplicates_removed:,} duplicate rows")
        if self.dropped_columns:
            lines.append(f"Dropped high-missing columns: {self.dropped_columns}")
        if self.imputed_columns:
            lines.append(f"Imputed {len(self.imputed_columns)} columns")
        if self.outliers_handled:
            total = sum(self.outliers_handled.values())
            lines.append(f"Capped/removed {total:,} outliers across {len(self.outliers_handled)} columns")
        if self.new_columns:
            lines.append(f"Created derived columns: {self.new_columns}")
        return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class CleaningAgent:
    """
    Cleans a DataFrame given a DatasetProfile.
    All operations are non-destructive — a copy of the DataFrame is returned.
    """

    def __init__(self, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> tuple[pd.DataFrame, CleaningReport]:
        """
        Apply full cleaning pipeline.

        Returns:
            (cleaned_df, report)
        """
        logger.info("CleaningAgent: starting pipeline...")
        df = df.copy()
        report = CleaningReport(original_shape=df.shape)

        df, report = self._drop_high_missing(df, profile, report)
        df, report = self._remove_duplicates(df, report)
        df, report = self._impute_missing(df, profile, report)
        df, report = self._handle_outliers(df, profile, report)
        df, report = self._engineer_features(df, profile, report)

        report.final_shape = df.shape
        logger.info(f"Cleaning complete. {report.summary()}")
        return df, report

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _drop_high_missing(
        self, df: pd.DataFrame, profile: DatasetProfile, report: CleaningReport
    ) -> tuple[pd.DataFrame, CleaningReport]:
        threshold = self.config.missing_threshold
        to_drop = [
            col
            for col, pct in profile.missing_pct.items()
            if pct / 100 > threshold
        ]
        if to_drop:
            df.drop(columns=to_drop, inplace=True)
            report.dropped_columns.extend(to_drop)
            report.steps.append(
                f"Dropped columns with >{threshold*100:.0f}% missing: {to_drop}"
            )
            logger.info(f"Dropped {len(to_drop)} high-missing columns: {to_drop}")
        return df, report

    def _remove_duplicates(
        self, df: pd.DataFrame, report: CleaningReport
    ) -> tuple[pd.DataFrame, CleaningReport]:
        if not self.config.remove_duplicates:
            return df, report
        n_before = len(df)
        df.drop_duplicates(inplace=True)
        removed = n_before - len(df)
        if removed:
            report.duplicates_removed = removed
            report.steps.append(f"Removed {removed:,} duplicate rows")
        return df, report

    def _impute_missing(
        self, df: pd.DataFrame, profile: DatasetProfile, report: CleaningReport
    ) -> tuple[pd.DataFrame, CleaningReport]:
        num_strategy = self.config.numerical_impute
        cat_strategy = self.config.categorical_impute

        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue

            if col in profile.numerical and col not in profile.categorical:
                fill_val = self._fill_value(df[col], num_strategy)
                if fill_val is not None:
                    df[col] = df[col].fillna(fill_val)
                    report.imputed_columns[col] = num_strategy
            elif col in profile.categorical or df[col].dtype == object:
                fill_val = self._fill_value(df[col], cat_strategy)
                if fill_val is not None:
                    df[col] = df[col].fillna(fill_val)
                    report.imputed_columns[col] = cat_strategy

        if report.imputed_columns:
            report.steps.append(
                f"Imputed missing values in {len(report.imputed_columns)} columns"
            )
        return df, report

    @staticmethod
    def _fill_value(series: pd.Series, strategy: ImpStrategy):
        if strategy == "mean":
            return series.mean()
        elif strategy == "median":
            return series.median()
        elif strategy == "mode":
            mode = series.mode()
            return mode.iloc[0] if not mode.empty else None
        return None  # "drop" handled separately

    def _handle_outliers(
        self, df: pd.DataFrame, profile: DatasetProfile, report: CleaningReport
    ) -> tuple[pd.DataFrame, CleaningReport]:
        method = self.config.outlier_method
        if method == "none":
            return df, report

        num_cols = [c for c in profile.numerical if c in df.columns]
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.config.iqr_multiplier * iqr
                upper = q3 + self.config.iqr_multiplier * iqr
            else:  # zscore
                mean, std = series.mean(), series.std()
                lower = mean - self.config.zscore_threshold * std
                upper = mean + self.config.zscore_threshold * std

            outlier_mask = (df[col] < lower) | (df[col] > upper)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                if self.config.cap_outliers:
                    df[col] = df[col].clip(lower=lower, upper=upper)
                else:
                    df = df[~outlier_mask]
                report.outliers_handled[col] = n_outliers

        if report.outliers_handled:
            total = sum(report.outliers_handled.values())
            report.steps.append(
                f"Handled {total:,} outliers via {method} method"
            )
        return df, report

    def _engineer_features(
        self, df: pd.DataFrame, profile: DatasetProfile, report: CleaningReport
    ) -> tuple[pd.DataFrame, CleaningReport]:
        if not self.config.parse_dates:
            return df, report

        new_cols: list[str] = []
        for col in profile.datetime:
            if col not in df.columns:
                continue
            try:
                dt = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if dt.notna().sum() < len(df) * 0.5:
                    continue
                df[col] = dt
                report.date_columns_parsed.append(col)
                if self.config.create_date_parts:
                    df[f"{col}_year"] = dt.dt.year
                    df[f"{col}_month"] = dt.dt.month
                    df[f"{col}_dayofweek"] = dt.dt.dayofweek
                    new_cols.extend(
                        [f"{col}_year", f"{col}_month", f"{col}_dayofweek"]
                    )
            except Exception as exc:
                logger.warning(f"Date parsing failed for {col}: {exc}")

        if new_cols:
            report.new_columns.extend(new_cols)
            report.steps.append(f"Created date-part features: {new_cols}")

        return df, report
