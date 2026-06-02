"""
FastAPI backend — REST API for the AI Data Analyst Agent.
Endpoints: upload, profile, clean, eda, insights, query, report, status.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from app.agents import AnalystPipeline, PipelineResult
from app.agents.query_agent import QueryAgent
from app.config import APP_NAME, APP_VERSION, DATA_DIR

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Production-grade AI Data Analyst Agent API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (use Redis in production)
_sessions: dict[str, PipelineResult] = {}


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    code: str = ""
    success: bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    business_context: str = Form(default=""),
    title: str = Form(default="Data Analysis Report"),
    use_llm: bool = Form(default=True),
) -> dict:
    """
    Upload a dataset file (CSV / Excel / JSON).
    Runs the full pipeline and returns a session_id for subsequent queries.
    """
    logger.info(f"Upload: {file.filename} ({file.content_type})")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    session_id = str(uuid.uuid4())

    try:
        pipeline = AnalystPipeline(use_llm=use_llm, save_charts=True)
        result = pipeline.run_from_file(
            raw, filename=file.filename, business_context=business_context, title=title
        )
        _sessions[session_id] = result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return {
        "session_id": session_id,
        "message": "Dataset analysed successfully",
        "rows": result.profile.rows,
        "columns": result.profile.columns,
    }


@app.get("/profile/{session_id}")
def get_profile(session_id: str) -> dict:
    result = _get_session(session_id)
    return result.profile.to_dict()


@app.get("/cleaning/{session_id}")
def get_cleaning_report(session_id: str) -> dict:
    result = _get_session(session_id)
    from dataclasses import asdict
    return asdict(result.cleaning_report)


@app.get("/eda/{session_id}")
def get_eda(session_id: str) -> dict:
    result = _get_session(session_id)
    return {
        "numerical_stats": result.eda_results.numerical_stats,
        "categorical_stats": result.eda_results.categorical_stats,
        "correlation_matrix": result.eda_results.correlation_matrix,
        "skewness": result.eda_results.skewness,
        "chart_paths": result.eda_results.chart_paths,
    }


@app.get("/insights/{session_id}")
def get_insights(session_id: str) -> dict:
    result = _get_session(session_id)
    from dataclasses import asdict
    return asdict(result.insight_report)


@app.post("/query", response_model=QueryResponse)
def query_dataset(req: QueryRequest) -> QueryResponse:
    result = _get_session(req.session_id)
    if result.query_agent is None:
        raise HTTPException(status_code=503, detail="Query agent not ready")
    qr = result.query_agent.ask(req.question)
    return QueryResponse(
        question=qr.question,
        answer=qr.answer,
        code=qr.code,
        success=qr.success,
    )


@app.get("/report/{session_id}")
def download_report(session_id: str, fmt: str = "html") -> FileResponse:
    result = _get_session(session_id)
    if fmt == "pdf" and result.pdf_report_path:
        path = result.pdf_report_path
        media = "application/pdf"
    else:
        path = result.html_report_path
        media = "text/html"
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=path, media_type=media, filename=Path(path).name)


@app.get("/sessions")
def list_sessions() -> dict:
    return {"sessions": list(_sessions.keys()), "count": len(_sessions)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> PipelineResult:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _sessions[session_id]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
