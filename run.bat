@echo off
REM CMD launcher
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
python app.py --config config.yaml
