@echo off
REM ---------------------------------------------------------------------------
REM Launch the Agentic AutoML interface.
REM
REM Invokes Streamlit through the virtual environment's own interpreter
REM (python -m streamlit) rather than the "streamlit" command. A bare
REM "streamlit run app.py" can pick up a different Streamlit that happens to be
REM earlier on PATH, which then fails to import this project's dependencies.
REM Going through the interpreter makes the launch independent of PATH order.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   No virtual environment found in this folder.
    echo   Create one and install the dependencies with:
    echo.
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting Agentic AutoML ...
echo Press Ctrl+C in this window to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
