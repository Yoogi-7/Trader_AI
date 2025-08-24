# Trader AI Screener (Windows 11, local)

Skaner 10m + MTF (1h/4h). Entry/SL/TP, sizing i dźwignia, backtest (trailing + time-stop), scheduler co 3 min, dashboard Streamlit oraz **ML – predykcja p(win)** trenowana na **ostatnich 2 latach**.

## 1) Instalacja
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
