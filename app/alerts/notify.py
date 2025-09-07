"""Simple alert notifiers (Discord/Telegram) using environment variables.

Env variables:
  - DISCORD_WEBHOOK_URL
  - TELEGRAM_BOT_TOKEN
  - TELEGRAM_CHAT_ID

Usage:
  from app.alerts.notify import notify_signal
  notify_signal(signal_dict)
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse

def _post_json(url: str, payload: dict) -> int:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
        return resp.getcode()

def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))

def send_discord(message: str) -> bool:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False
    try:
        code = _post_json(url, {"content": message})
        return 200 <= code < 300
    except Exception:
        return False

def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        code = _post_json(api_url, payload)
        return 200 <= code < 300
    except Exception:
        return False

def notify_signal(sig: dict) -> dict:
    """Send a compact alert to both channels if configured. Returns result flags."""
    sym = sig.get("symbol")
    tf = sig.get("timeframe")
    side = sig.get("direction")
    p_hit = sig.get("p_hit")
    rr = sig.get("rr")
    entry = sig.get("entry")
    sl = sig.get("sl")
    tp = sig.get("tp")

    msg = (
        f"<b>Signal</b> {sym} [{tf}] {side}\n"
        f"Entry: {entry} | SL: {sl} | TP: {tp}\n"
        f"p_hit: {p_hit} | RR: {rr}"
    )
    ok_discord = send_discord(message=msg.replace("<b>", "**").replace("</b>", "**"))
    ok_telegram = send_telegram(message=msg)
    return {"discord": ok_discord, "telegram": ok_telegram}
