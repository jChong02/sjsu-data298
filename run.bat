@echo off
echo Starting Medical LLM XAI Toolkit...
cd /d "%~dp0"
python -m streamlit run app/main.py
pause
