@echo off
cd /d "%~dp0"
if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" "%~dp0smart_compressor.py"
) else (
    python "%~dp0smart_compressor.py"
)
