@echo off
setlocal
cd /d "%~dp0"

echo [1/5] Checking Python launcher...
where py >nul 2>nul
if not errorlevel 1 (
  set "PY_CMD=py"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3.10 or newer was not found. Install 64-bit Python from python.org and enable the PATH option.
    pause
    exit /b 1
  )
  set "PY_CMD=python"
)

%PY_CMD% -c "import sys, struct; print('Using Python', sys.version); print('Python architecture:', struct.calcsize('P')*8, 'bit'); sys.exit(0 if sys.version_info >= (3,10) and struct.calcsize('P')*8 == 64 else 1)"
if errorlevel 1 (
  echo This project requires 64-bit Python 3.10 or newer. Install the matching Python version and retry.
  goto :error
)
if not exist ".venv\Scripts\python.exe" (
  echo [2/5] Creating virtual environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto :error
)

echo [3/5] Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [4/5] Downloading dataset...
.venv\Scripts\python.exe download_data.py
if errorlevel 1 goto :error

echo [5/5] Training and evaluating models...
.venv\Scripts\python.exe train.py
if errorlevel 1 goto :error

echo.
echo Setup and training completed. Run run_app.bat to open the dashboard.
pause
exit /b 0

:error
echo.
echo Setup stopped because a command failed. Read the error above and share it for troubleshooting.
pause
exit /b 1
