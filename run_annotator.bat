@echo off
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m semantic_dataset annotate
) else (
    python -m semantic_dataset annotate
)
if errorlevel 1 pause
