# 📊 AI Data Analyst Agent

A production-quality, modular AI data analyst that behaves like a human senior analyst.
Upload any dataset and get automatic EDA, AI-generated insights, natural language Q&A,
interactive dashboards, and downloadable reports.

---

## ✨ Features

| Module | What it does |
|--------|-------------|
| 🔍 Dataset Understanding | Auto-detects schema, types, missing values, potential targets |
| 🧹 Data Cleaning | Handles missing values, duplicates, outliers, date features |
| 📈 EDA | Descriptive stats, distributions, correlations, 8+ chart types |
| 💡 Insight Generation | LLM-powered business insights, anomaly detection, recommendations |
| 💬 NL Query Agent | Ask questions in plain English — computed via generated pandas code |
| 📊 Streamlit Dashboard | Interactive 5-tab UI with all outputs |
| 📄 Report Generator | Downloadable HTML (+ PDF via WeasyPrint) report |
| ⚡ FastAPI Backend | REST API for all pipeline operations |

---

## 🚀 Quick Start

### 1. Clone & configure

```bash
git clone <repo-url>
cd data-analyst-agent
cp .env.example .env
# Edit .env and add your GROQ_API_KEY or OPENAI_API_KEY
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit dashboard

```bash
streamlit run dashboard/streamlit_app.py
```
Open http://localhost:8501

### 4. Run the FastAPI backend (optional)

```bash
uvicorn app.main:app --reload --port 8000
```
API docs at http://localhost:8000/docs

---

## 🐳 Docker

```bash
# Build and run everything
docker-compose up --build

# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000/docs
```

---

## 📁 Project Structure

```
data-analyst-agent/
├── app/
│   ├── agents/
│   │   ├── __init__.py          # AnalystPipeline orchestrator
│   │   ├── data_understanding.py  # Module 1 — dataset profiler
│   │   ├── cleaning_agent.py      # Module 2 — data cleaner
│   │   ├── eda_agent.py           # Module 3 — EDA + charts
│   │   ├── insight_generator.py   # Module 4 — LLM insights
│   │   ├── query_agent.py         # Module 5 — NL query engine
│   │   └── report_generator.py    # Module 7 — HTML/PDF reports
│   ├── utils/
│   │   ├── data_loader.py         # CSV / Excel / JSON loader
│   │   └── llm_client.py          # Groq / OpenAI wrapper
│   ├── tests/
│   │   └── test_agents.py         # pytest unit tests
│   ├── config.py                  # Settings / env vars
│   └── main.py                    # FastAPI application
├── dashboard/
│   └── streamlit_app.py           # Module 6 — Streamlit UI
├── data/                          # Input datasets
├── reports/                       # Generated HTML/PDF reports
├── visualizations/                # Saved chart HTML files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/upload` | Upload dataset, run full pipeline, get session_id |
| `GET`  | `/profile/{session_id}` | Dataset profile |
| `GET`  | `/cleaning/{session_id}` | Cleaning report |
| `GET`  | `/eda/{session_id}` | EDA statistics |
| `GET`  | `/insights/{session_id}` | AI insights |
| `POST` | `/query` | Ask a natural language question |
| `GET`  | `/report/{session_id}?fmt=html` | Download report |

### Upload example

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@customer_churn.csv" \
  -F "business_context=Telecom churn analysis" \
  -F "use_llm=true"
```

### Query example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "question": "Which region has the highest average income?"}'
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `LLM_MODEL` | `llama3-70b-8192` | Model name |
| `GROQ_API_KEY` | — | Your Groq API key |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `MAX_INSIGHTS` | `15` | Max insights to generate |
| `MAX_CHART_ROWS` | `10000` | Max rows sampled for charts |

---

## 🧪 Running Tests

```bash
pytest app/tests/ -v
```

---

## 🛠️ Supported Input Formats

- **CSV** — standard comma-separated files
- **Excel** — `.xlsx` and `.xls`  
- **JSON** — records-oriented JSON arrays

---

