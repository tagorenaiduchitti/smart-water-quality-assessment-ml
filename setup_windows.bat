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
    echo Python 3.10 or newer was not found. Install 64-bit Python from python.org and enable PATH.
    goto :error
  )
  set "PY_CMD=python"
)

%PY_CMD% -c "import sys, struct; print('Using Python', sys.version); print('Python architecture:', struct.calcsize('P')*8, 'bit'); sys.exit(0 if sys.version_info >= (3,10) and struct.calcsize('P')*8 == 64 else 1)"
if errorlevel 1 (
  echo This project requires 64-bit Python 3.10 or newer.
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

echo [4/5] Checking for the Kaggle dataset...
.venv\Scripts\python.exe download_data.py
if errorlevel 1 goto :error

echo [5/5] Training and evaluating models...
.venv\Scripts\python.exe train.py
if errorlevel 1 goto :error

echo.
echo Setup and training completed. Run run_app.bat or press F5 in VS Code.
pause
exit /b 0

:error
echo.
echo Setup stopped. Confirm that data\waterQuality1.csv is present and review the error above.
pause
exit /b 1
