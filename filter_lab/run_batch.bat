@echo off
setlocal
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" "filter_lab.py" --batch %*
) else (
  python "filter_lab.py" --batch %*
)
pause
