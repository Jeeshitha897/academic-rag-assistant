@echo off
title Academic Doubt-Solving RAG Chatbot - Lanka Jeeshita
echo ===================================================================
echo Starting Academic Doubt-Solving RAG Chatbot
echo Student: Lanka Jeeshita ^| Supervisor: Dr. P. Sudhakar
echo ===================================================================
echo.
cd /d "%~dp0"
py -3.13 -m streamlit run app.py
pause
