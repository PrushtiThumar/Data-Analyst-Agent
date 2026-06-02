"""
Module 7 — Report Generator
Produces a standalone HTML report and optionally a PDF.
Includes: dataset overview, data quality, EDA stats, charts, insights, recommendations.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader
from loguru import logger

from app.agents.cleaning_agent import CleaningReport
from app.agents.data_understanding import DatasetProfile
from app.agents.eda_agent import EDAResults
from app.agents.insight_generator import InsightReport
from app.config import APP_NAME, APP_VERSION, REPORTS_DIR


# ── HTML Template ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  :root {
    --ink: #1a1a2e;
    --accent: #4361ee;
    --accent2: #3a0ca3;
    --green: #06d6a0;
    --amber: #ffd166;
    --red: #ef476f;
    --bg: #f8f9ff;
    --card: #ffffff;
    --border: #e2e8f0;
    --muted: #64748b;
    --font-heading: 'Georgia', serif;
    --font-body: 'Segoe UI', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: var(--font-body); background: var(--bg); color: var(--ink); line-height: 1.6; }

  /* Header */
  .report-header {
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    color: white;
    padding: 3rem 2rem 2rem;
    text-align: center;
  }
  .report-header h1 { font-family: var(--font-heading); font-size: 2.4rem; margin-bottom: 0.5rem; }
  .report-header p  { opacity: 0.85; font-size: 1rem; }
  .report-meta { margin-top: 1.2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; }
  .report-meta span { background: rgba(255,255,255,.15); padding: .3rem .9rem; border-radius: 2rem; font-size: .85rem; }

  /* Layout */
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
  .section { margin-bottom: 2.5rem; }
  .section-title {
    font-family: var(--font-heading);
    font-size: 1.4rem;
    color: var(--accent2);
    border-left: 4px solid var(--accent);
    padding-left: .75rem;
    margin-bottom: 1.2rem;
  }

  /* Cards */
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,.05); }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }
  .stat-card .value { font-size: 1.8rem; font-weight: 700; color: var(--accent); }
  .stat-card .label { font-size: .78rem; color: var(--muted); margin-top: .2rem; text-transform: uppercase; letter-spacing: .05em; }

  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: .88rem; }
  th { background: var(--accent); color: white; padding: .55rem .8rem; text-align: left; }
  td { padding: .5rem .8rem; border-bottom: 1px solid var(--border); }
  tr:hover td { background: #f0f4ff; }

  /* Insight list */
  .insight-item {
    padding: .8rem 1rem;
    margin-bottom: .6rem;
    background: var(--card);
    border-left: 3px solid var(--green);
    border-radius: 0 8px 8px 0;
    font-size: .92rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }
  .recommendation-item {
    padding: .8rem 1rem;
    margin-bottom: .6rem;
    background: #fffbeb;
    border-left: 3px solid var(--amber);
    border-radius: 0 8px 8px 0;
    font-size: .92rem;
  }
  .anomaly-item {
    padding: .8rem 1rem;
    margin-bottom: .6rem;
    background: #fff1f5;
    border-left: 3px solid var(--red);
    border-radius: 0 8px 8px 0;
    font-size: .92rem;
  }
  .quality-item {
    padding: .7rem 1rem;
    margin-bottom: .5rem;
    background: #f8faff;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: .88rem;
    color: var(--muted);
  }

  /* Chart container */
  .chart-container { margin-bottom: 1.5rem; }

  /* Footer */
  .report-footer { text-align: center; padding: 2rem; color: var(--muted); font-size: .82rem; border-top: 1px solid var(--border); margin-top: 2rem; }

  /* Badges */
  .badge { display: inline-block; padding: .2rem .65rem; border-radius: 2rem; font-size: .75rem; font-weight: 600; }
  .badge-blue { background: #dbeafe; color: #1d4ed8; }
  .badge-green { background: #d1fae5; color: #065f46; }
  .badge-amber { background: #fef3c7; color: #92400e; }
  .badge-red { background: #fee2e2; color: #991b1b; }
</style>
</head>
<body>

<div class="report-header">
  <h1>📊 {{ title }}</h1>
  <p>{{ business_context }}</p>
  <div class="report-meta">
    <span>📅 {{ generated_at }}</span>
    <span>🔢 {{ profile.rows | format_num }} rows</span>
    <span>📋 {{ profile.columns }} columns</span>
    <span>⚙️ {{ app_name }} v{{ app_version }}</span>
  </div>
</div>

<div class="container">

  <!-- 1. Dataset Overview -->
  <section class="section">
    <h2 class="section-title">1. Dataset Overview</h2>
    <div class="stat-grid">
      <div class="stat-card"><div class="value">{{ profile.rows | format_num }}</div><div class="label">Total Rows</div></div>
      <div class="stat-card"><div class="value">{{ profile.columns }}</div><div class="label">Columns</div></div>
      <div class="stat-card"><div class="value">{{ profile.numerical | length }}</div><div class="label">Numerical</div></div>
      <div class="stat-card"><div class="value">{{ profile.categorical | length }}</div><div class="label">Categorical</div></div>
      <div class="stat-card"><div class="value">{{ profile.datetime | length }}</div><div class="label">Datetime</div></div>
      <div class="stat-card"><div class="value">{{ profile.duplicate_rows | format_num }}</div><div class="label">Duplicates</div></div>
      <div class="stat-card"><div class="value">{{ profile.memory_mb }} MB</div><div class="label">Memory</div></div>
    </div>

    {% if profile.summary_text %}
    <div class="card" style="margin-top:1.2rem;">
      <p>{{ profile.summary_text }}</p>
    </div>
    {% endif %}
  </section>

  <!-- 2. Data Quality -->
  <section class="section">
    <h2 class="section-title">2. Data Quality</h2>

    {% if cleaning_report %}
    <div class="card">
      <table>
        <thead><tr><th>Cleaning Step</th></tr></thead>
        <tbody>
          {% for step in cleaning_report.steps %}
          <tr><td>✅ {{ step }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}

    {% if insight_report.data_quality_notes %}
    <div style="margin-top:.8rem;">
      {% for note in insight_report.data_quality_notes %}
      <div class="quality-item">⚠️ {{ note }}</div>
      {% endfor %}
    </div>
    {% endif %}

    <!-- Missing values table -->
    {% if missing_cols %}
    <div class="card" style="margin-top:1rem; overflow-x:auto;">
      <table>
        <thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr></thead>
        <tbody>
          {% for col, pct in missing_cols %}
          <tr>
            <td>{{ col }}</td>
            <td>{{ profile.missing[col] | format_num }}</td>
            <td>
              <span class="badge {% if pct > 30 %}badge-red{% elif pct > 10 %}badge-amber{% else %}badge-green{% endif %}">
                {{ pct }}%
              </span>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}
  </section>

  <!-- 3. Statistical Summary -->
  <section class="section">
    <h2 class="section-title">3. Statistical Summary</h2>
    {% if eda_results.numerical_stats %}
    <div style="overflow-x:auto;">
    <table>
      <thead>
        <tr><th>Column</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Skewness</th></tr>
      </thead>
      <tbody>
        {% for col, s in eda_results.numerical_stats.items() %}
        <tr>
          <td><strong>{{ col }}</strong></td>
          <td>{{ s.mean }}</td>
          <td>{{ s.median }}</td>
          <td>{{ s.std }}</td>
          <td>{{ s.min }}</td>
          <td>{{ s.max }}</td>
          <td>
            <span class="badge {% if s.skewness | abs > 1 %}badge-amber{% else %}badge-green{% endif %}">
              {{ s.skewness }}
            </span>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
    {% endif %}
  </section>

  <!-- 4. Visualizations -->
  {% if chart_htmls %}
  <section class="section">
    <h2 class="section-title">4. Visualizations</h2>
    {% for chart_html in chart_htmls %}
    <div class="chart-container card">
      {{ chart_html }}
    </div>
    {% endfor %}
  </section>
  {% endif %}

  <!-- 5. Key Insights -->
  <section class="section">
    <h2 class="section-title">5. Key Insights</h2>
    {% for insight in insight_report.insights %}
    <div class="insight-item">💡 {{ insight }}</div>
    {% endfor %}

    {% if insight_report.anomalies %}
    <h3 style="margin-top:1.2rem; margin-bottom:.6rem; font-size:1.05rem; color:var(--red);">Anomalies</h3>
    {% for item in insight_report.anomalies %}
    <div class="anomaly-item">🔍 {{ item }}</div>
    {% endfor %}
    {% endif %}
  </section>

  <!-- 6. Recommendations -->
  {% if insight_report.recommendations %}
  <section class="section">
    <h2 class="section-title">6. Recommendations</h2>
    {% for rec in insight_report.recommendations %}
    <div class="recommendation-item">📌 {{ rec }}</div>
    {% endfor %}
  </section>
  {% endif %}

</div>

<div class="report-footer">
  Generated by <strong>{{ app_name }}</strong> v{{ app_version }} · {{ generated_at }}
</div>

</body>
</html>
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Generates a standalone HTML report (and optionally PDF).
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or REPORTS_DIR
        self._env = Environment(loader=BaseLoader())
        self._env.filters["format_num"] = lambda v: f"{int(v):,}"
        self._env.filters["abs"] = abs
        self._template = self._env.from_string(_HTML_TEMPLATE)

    # ── Public ────────────────────────────────────────────────────────────────

    def generate_html(
        self,
        profile: DatasetProfile,
        cleaning_report: CleaningReport,
        eda_results: EDAResults,
        insight_report: InsightReport,
        business_context: str = "",
        title: str = "Data Analysis Report",
    ) -> str:
        """
        Render the HTML report and save it to disk.

        Returns:
            Path to the saved HTML file.
        """
        logger.info("ReportGenerator: building HTML report...")

        chart_htmls = self._embed_charts(eda_results.chart_figures)
        missing_cols = [
            (col, pct)
            for col, pct in sorted(
                profile.missing_pct.items(), key=lambda x: x[1], reverse=True
            )
            if pct > 0
        ][:20]

        html = self._template.render(
            title=title,
            business_context=business_context or "Automated data analysis",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            profile=profile,
            cleaning_report=cleaning_report,
            eda_results=eda_results,
            insight_report=insight_report,
            chart_htmls=chart_htmls,
            missing_cols=missing_cols,
            app_name=APP_NAME,
            app_version=APP_VERSION,
        )

        path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved: {path}")
        return str(path)

    def generate_pdf(self, html_path: str) -> str | None:
        """
        Convert a saved HTML report to PDF using WeasyPrint.
        Returns the PDF path or None if WeasyPrint is unavailable.
        """
        try:
            from weasyprint import HTML as WP_HTML
            pdf_path = html_path.replace(".html", ".pdf")
            WP_HTML(filename=html_path).write_pdf(pdf_path)
            logger.info(f"PDF report saved: {pdf_path}")
            return pdf_path
        except ImportError:
            logger.warning("WeasyPrint not installed; skipping PDF generation.")
            return None
        except Exception as exc:
            logger.error(f"PDF generation failed: {exc}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _embed_charts(figs: list[go.Figure]) -> list[str]:
        """Convert Plotly figures to inline HTML fragments."""
        html_parts: list[str] = []
        for fig in figs:
            try:
                html_parts.append(
                    fig.to_html(full_html=False, include_plotlyjs="cdn")
                )
            except Exception as exc:
                logger.warning(f"Failed to embed chart: {exc}")
        return html_parts
