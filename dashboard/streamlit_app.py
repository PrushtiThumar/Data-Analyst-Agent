"""
Module 6 — Streamlit Dashboard
Full interactive UI for the AI Data Analyst Agent.
Tabs: Overview · EDA · Insights · Chat · Report
"""

from __future__ import annotations

import io
import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from app.agents import AnalystPipeline, PipelineResult
from app.agents.cleaning_agent import CleaningConfig
from app.config import APP_NAME, APP_VERSION

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }
[data-testid="stSidebar"] .stMarkdown h2 { color: #a78bfa !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #f8f9ff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: #4361ee !important; font-weight: 700; }

/* Tabs */
[data-testid="stTabs"] [role="tab"] { font-size: 1rem; font-weight: 500; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #4361ee;
    border-bottom: 2px solid #4361ee;
}

/* Insight card */
.insight-card {
    background: #f0fdf4;
    border-left: 4px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.93rem;
    line-height: 1.6;
}
.rec-card {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.93rem;
}
.anomaly-card {
    background: #fff1f5;
    border-left: 4px solid #ef4444;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.93rem;
}
.chat-user {
    background: #eff6ff;
    border-radius: 12px 12px 4px 12px;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    text-align: right;
}
.chat-bot {
    background: #f0fdf4;
    border-radius: 12px 12px 12px 4px;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0;
    max-width: 80%;
    border-left: 3px solid #10b981;
}
.code-pill {
    font-family: 'Courier New', monospace;
    background: #1e293b;
    color: #7dd3fc;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    font-size: 0.82rem;
    display: block;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "result": None,
        "chat_history": [],
        "pipeline_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[bytes | None, str, str, CleaningConfig, bool]:
    st.sidebar.markdown("## 📊 AI Data Analyst")
    st.sidebar.markdown(f"*v{APP_VERSION}*")
    st.sidebar.markdown("---")

    uploaded = st.sidebar.file_uploader(
        "Upload Dataset", type=["csv", "xlsx", "xls", "json"],
        help="CSV, Excel, or JSON"
    )
    file_bytes = uploaded.read() if uploaded else None
    filename = uploaded.name if uploaded else ""

    st.sidebar.markdown("### ⚙️ Settings")
    business_context = st.sidebar.text_area(
        "Business Context",
        placeholder="e.g. Telecom customer churn analysis for Q4 2024",
        height=80,
    )
    report_title = st.sidebar.text_input("Report Title", value="Data Analysis Report")

    st.sidebar.markdown("### 🧹 Cleaning Options")
    num_strategy = st.sidebar.selectbox(
        "Numerical Imputation", ["median", "mean", "mode"], index=0
    )
    cat_strategy = st.sidebar.selectbox(
        "Categorical Imputation", ["mode", "mean", "median"], index=0
    )
    outlier_method = st.sidebar.selectbox(
        "Outlier Detection", ["iqr", "zscore", "none"], index=0
    )
    use_llm = st.sidebar.toggle("Enable AI Insights (LLM)", value=True)

    cleaning_config = CleaningConfig(
        numerical_impute=num_strategy,
        categorical_impute=cat_strategy,
        outlier_method=outlier_method,
    )

    run_btn = st.sidebar.button(
        "🚀 Run Analysis", type="primary", use_container_width=True,
        disabled=file_bytes is None,
    )

    if file_bytes and run_btn:
        return file_bytes, filename, business_context, cleaning_config, use_llm, report_title, True

    return file_bytes, filename, business_context, cleaning_config, use_llm, report_title, False


# ── Tabs ──────────────────────────────────────────────────────────────────────

def render_overview(result: PipelineResult):
    p = result.profile
    cr = result.cleaning_report

    st.markdown("## 📋 Dataset Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{p.rows:,}")
    c2.metric("Columns", str(p.columns))
    c3.metric("Numerical", len(p.numerical))
    c4.metric("Categorical", len(p.categorical))
    c5.metric("Duplicates", f"{p.duplicate_rows:,}")
    c6.metric("Memory", f"{p.memory_mb} MB")

    if p.summary_text:
        st.markdown("### 🤖 AI Summary")
        st.info(p.summary_text)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📝 Column Types")
        type_data = {"Type": [], "Columns": []}
        for label, cols in [
            ("Numerical", p.numerical),
            ("Categorical", p.categorical),
            ("Datetime", p.datetime),
            ("Boolean", p.boolean),
        ]:
            if cols:
                type_data["Type"].append(label)
                type_data["Columns"].append(", ".join(cols[:5]) + ("..." if len(cols) > 5 else ""))
        st.dataframe(pd.DataFrame(type_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🔍 Missing Values")
        missing = {k: v for k, v in p.missing_pct.items() if v > 0}
        if missing:
            miss_df = pd.DataFrame(
                [(k, v) for k, v in sorted(missing.items(), key=lambda x: x[1], reverse=True)],
                columns=["Column", "Missing %"],
            )
            fig = px.bar(
                miss_df, x="Missing %", y="Column", orientation="h",
                color="Missing %", color_continuous_scale="Reds",
                template="plotly_white", height=300,
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(l=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    st.markdown("### 🧹 Cleaning Summary")
    if cr.steps:
        for step in cr.steps:
            st.markdown(f"✅ {step}")
    else:
        st.info("No cleaning steps were required.")

    if p.potential_targets:
        st.markdown("### 🎯 Potential Target Variables")
        st.write(", ".join(f"`{t}`" for t in p.potential_targets))

    st.markdown("### 📊 Raw Data Preview")
    st.dataframe(result.df_clean.head(20), use_container_width=True)


def render_eda(result: PipelineResult):
    eda = result.eda_results
    st.markdown("## 🔬 Exploratory Data Analysis")

    if eda.numerical_stats:
        st.markdown("### 📈 Numerical Statistics")
        rows = []
        for col, s in eda.numerical_stats.items():
            rows.append({
                "Column": col, "Mean": s["mean"], "Median": s["median"],
                "Std": s["std"], "Min": s["min"], "Max": s["max"],
                "Skewness": s["skewness"], "Kurtosis": s["kurtosis"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Visualizations
    if eda.chart_figures:
        st.markdown("### 📊 Visualizations")
        for fig in eda.chart_figures:
            st.plotly_chart(fig, use_container_width=True)
    elif eda.chart_paths:
        st.info(f"Charts saved to: {eda.chart_paths[0]}")

    if eda.categorical_stats:
        st.markdown("### 🏷️ Categorical Features")
        for col, stats in list(eda.categorical_stats.items())[:6]:
            with st.expander(f"📌 {col} — {stats['unique']} unique values"):
                vc = stats.get("top_values", {})
                if vc:
                    df_vc = pd.DataFrame(
                        list(vc.items()), columns=["Value", "Count"]
                    )
                    fig = px.bar(
                        df_vc, x="Count", y="Value", orientation="h",
                        template="plotly_white", height=max(200, 35 * len(df_vc)),
                    )
                    fig.update_layout(margin=dict(l=0))
                    st.plotly_chart(fig, use_container_width=True)


def render_insights(result: PipelineResult):
    ins = result.insight_report
    st.markdown("## 💡 Insights & Recommendations")

    if ins.insights:
        st.markdown("### Key Insights")
        for i, insight in enumerate(ins.insights, 1):
            st.markdown(
                f'<div class="insight-card">💡 <strong>#{i}</strong> {insight}</div>',
                unsafe_allow_html=True,
            )

    if ins.recommendations:
        st.markdown("### 📌 Recommendations")
        for rec in ins.recommendations:
            st.markdown(
                f'<div class="rec-card">📌 {rec}</div>', unsafe_allow_html=True
            )

    if ins.anomalies:
        st.markdown("### 🔍 Anomalies")
        for a in ins.anomalies:
            st.markdown(
                f'<div class="anomaly-card">⚠️ {a}</div>', unsafe_allow_html=True
            )

    if ins.key_drivers:
        st.markdown("### 🎯 Key Drivers")
        st.write("Columns most likely to drive outcomes: " + ", ".join(f"`{d}`" for d in ins.key_drivers))

    if ins.data_quality_notes:
        st.markdown("### 🧪 Data Quality Notes")
        for note in ins.data_quality_notes:
            st.warning(note)


def render_chat(result: PipelineResult):
    st.markdown("## 💬 Ask Questions About Your Data")
    st.caption("Ask anything in plain English — the agent will compute an answer.")

    if result.query_agent is None:
        st.error("Query agent not available.")
        return

    # Sample questions
    st.markdown("**Try asking:**")
    sample_questions = [
        "What is the average value of each numerical column?",
        "Which category appears most frequently?",
        "Show the top 5 rows sorted by the first numerical column",
        "What percentage of rows have missing values?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        if cols[i % 2].button(f"📝 {q}", key=f"sample_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                qr = result.query_agent.ask(q)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": qr.answer,
                "code": qr.code,
                "success": qr.success,
            })

    st.markdown("---")

    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">🙋 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("code"):
                with st.expander("📄 View generated code"):
                    st.code(msg["code"], language="python")

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question", placeholder="e.g. Which region has the highest revenue?"
        )
        submitted = st.form_submit_button("Ask ➤", type="primary")

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Computing answer..."):
            qr = result.query_agent.ask(user_input)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": qr.answer,
            "code": qr.code,
            "success": qr.success,
        })
        st.rerun()


def render_report(result: PipelineResult):
    st.markdown("## 📄 Download Report")

    col1, col2 = st.columns(2)

    if result.html_report_path and Path(result.html_report_path).exists():
        html_bytes = Path(result.html_report_path).read_bytes()
        col1.download_button(
            "⬇️ Download HTML Report",
            data=html_bytes,
            file_name=Path(result.html_report_path).name,
            mime="text/html",
            use_container_width=True,
        )
        with st.expander("👁️ Preview Report"):
            st.components.v1.html(
                html_bytes.decode("utf-8", errors="replace"),
                height=800,
                scrolling=True,
            )
    else:
        col1.info("HTML report will appear here after analysis.")

    if result.pdf_report_path and Path(result.pdf_report_path).exists():
        col2.download_button(
            "⬇️ Download PDF Report",
            data=Path(result.pdf_report_path).read_bytes(),
            file_name=Path(result.pdf_report_path).name,
            mime="application/pdf",
            use_container_width=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init_state()

    sidebar_return = render_sidebar()
    file_bytes, filename, business_context, cleaning_config, use_llm, report_title, should_run = sidebar_return

    # Run pipeline
    if should_run and file_bytes:
        st.session_state.chat_history = []
        with st.spinner("🔍 Running full analysis pipeline... this may take a moment"):
            try:
                pipeline = AnalystPipeline(
                    use_llm=use_llm,
                    cleaning_config=cleaning_config,
                    save_charts=True,
                )
                result = pipeline.run_from_file(
                    file_bytes,
                    filename=filename,
                    business_context=business_context,
                    title=report_title,
                )
                st.session_state.result = result
                st.session_state.pipeline_run = True
                st.success("✅ Analysis complete!")
            except Exception as exc:
                st.error(f"❌ Analysis failed: {exc}")
                logger.exception(exc)

    # Render tabs
    if st.session_state.result is not None:
        result: PipelineResult = st.session_state.result
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📋 Overview", "🔬 EDA", "💡 Insights", "💬 Chat", "📄 Report"]
        )
        with tab1:
            render_overview(result)
        with tab2:
            render_eda(result)
        with tab3:
            render_insights(result)
        with tab4:
            render_chat(result)
        with tab5:
            render_report(result)
    else:
        # Landing page
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
          <h1 style="font-size:2.8rem; color:#1a1a2e;">📊 AI Data Analyst Agent</h1>
          <p style="font-size:1.2rem; color:#64748b; max-width:600px; margin:1rem auto;">
            Upload a CSV, Excel, or JSON dataset and let the AI analyst automatically
            profile, clean, analyze, and generate insights for you.
          </p>
          <div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-top:2rem;">
            <div style="background:#f0f4ff; border-radius:12px; padding:1.2rem 1.8rem; min-width:140px;">
              <div style="font-size:2rem;">🔍</div>
              <div style="font-weight:600; margin-top:.4rem;">Auto EDA</div>
            </div>
            <div style="background:#f0fff4; border-radius:12px; padding:1.2rem 1.8rem; min-width:140px;">
              <div style="font-size:2rem;">💡</div>
              <div style="font-weight:600; margin-top:.4rem;">AI Insights</div>
            </div>
            <div style="background:#fff7ed; border-radius:12px; padding:1.2rem 1.8rem; min-width:140px;">
              <div style="font-size:2rem;">💬</div>
              <div style="font-weight:600; margin-top:.4rem;">NL Chat</div>
            </div>
            <div style="background:#fdf4ff; border-radius:12px; padding:1.2rem 1.8rem; min-width:140px;">
              <div style="font-size:2rem;">📄</div>
              <div style="font-weight:600; margin-top:.4rem;">Reports</div>
            </div>
          </div>
          <p style="color:#94a3b8; margin-top:2.5rem;">← Upload a dataset in the sidebar to begin</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
