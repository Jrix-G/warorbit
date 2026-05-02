@echo off
setlocal
cd /d "%~dp0"
set WARORBIT_SMOKE_MINUTES=5
set WARORBIT_SMOKE_EVAL_EPISODES=16
set WARORBIT_SMOKE_EVAL_SLICES=10
python -u "%~dp0lanceur_20min.py"
endlocal
