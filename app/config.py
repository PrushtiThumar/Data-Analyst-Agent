"""
Configuration and settings for the AI Data Analyst Agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
REPORTS_DIR = BASE_DIR / os.getenv("REPORTS_DIR", "reports")
VIZ_DIR = BASE_DIR / os.getenv("VISUALIZATIONS_DIR", "visualizations")

for _d in [DATA_DIR, REPORTS_DIR, VIZ_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-70b-8192")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── App ───────────────────────────────────────────────────────────────────────
APP_NAME: str = os.getenv("APP_NAME", "AI Data Analyst Agent")
APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

MAX_INSIGHTS: int = int(os.getenv("MAX_INSIGHTS", "15"))
MAX_CHART_ROWS: int = int(os.getenv("MAX_CHART_ROWS", "10000"))
