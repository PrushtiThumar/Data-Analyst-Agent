#!/bin/bash
# Start FastAPI in background, then run Streamlit in foreground
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
