@echo off
echo Starting LLM XAI Toolkit...
cd /d "%~dp0"
python -m streamlit run app/main.py
pause
