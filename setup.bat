@echo off
echo --------------------------------------
echo Setting up ML Project Environment
echo --------------------------------------

REM Check Python 3.11
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 not found.
    echo Please install Python 3.11.
    pause
    exit /b
)

echo Python 3.11 detected.

REM Create venv
if not exist ".venv" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
) else (
    echo .venv already exists. Skipping creation.
)

echo Activating virtual environment...
call .\.venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo --------------------------------------
echo Setup Complete!
echo Run: streamlit run app.py
echo --------------------------------------

pause
