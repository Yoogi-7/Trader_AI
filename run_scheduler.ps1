# Uruchamia skaner co X minut używając APScheduler
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
if (-not (Test-Path .\.venv)) { py -3.11 -m venv .venv }
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe scheduler_run.py --config config.yaml
