from __future__ import annotations
import requests
from datetime import datetime, timezone

def _tg_enabled(cfg: dict) -> bool:
    tg = cfg.get("notify", {}).get("telegram", {})
    return bool(tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"))

def _tg_send(cfg: dict, text: str):
    tg = cfg["notify"]["telegram"]
    url = f"https://api.telegram.org/bot{tg['bot_token']}/sendMessage"
    data = {
        "chat_id": tg["chat_id"],
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": 1,
    }
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"[TG] send error: {e}")

def fmt_price(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")

def fmt_ts_ms(ms: int) -> str:
    dt = datetime.fromtimestamp(ms/1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

def message_new_signal(exch, sym, tf, sig: dict, p: float | None = None, ev: float | None = None) -> str:
    dir_emoji = "🟢 LONG" if sig["direction"] == "long" else "🔴 SHORT"
    parts = [
        f"<b>{dir_emoji}</b>  <b>{sym}</b>  <code>{tf}</code>  ({exch})",
        f"⏱ {fmt_ts_ms(int(sig['ts_ms']))}",
        f"Entry: <code>{fmt_price(sig['entry'])}</code>",
        f"SL:    <code>{fmt_price(sig['sl'])}</code>",
        f"TP1:   <code>{fmt_price(sig['tp1'])}</code>   TP2: <code>{fmt_price(sig['tp2'])}</code>",
        f"Lev: <b>{sig['leverage']:.1f}x</b>   Risk: <b>{sig['risk_pct']:.2f}%</b>"
    ]
    if p is not None:
        parts.append(f"p(win): <b>{p*100:.1f}%</b>")
    if ev is not None:
        parts.append(f"EV: <b>{ev:.2f}%</b>")
    return "\n".join(parts)

def message_closed(sym, tf, row) -> str:
    status = str(row["status"])
    emoji = "✅ TP" if status in ("TP","TP1_TRAIL") else "❌ SL"
    parts = [
        f"<b>{emoji}</b>  <b>{sym}</b>  <code>{tf}</code>",
        f"⏱ {fmt_ts_ms(int(row['closed_ts_ms']))}",
        f"Exit: <code>{fmt_price(float(row['exit_price'] or 0.0))}</code>",
        f"PnL: <b>{float(row['pnl_usd'] or 0.0):.2f} USD</b>  ({float(row['pnl_pct'] or 0.0):.2f}%)",
    ]
    return "\n".join(parts)

def send_new_signal(cfg: dict, exch, sym, tf, sig: dict, p: float | None, ev: float | None):
    if not _tg_enabled(cfg): return
    _tg_send(cfg, message_new_signal(exch, sym, tf, sig, p, ev))

def send_closed_trade(cfg: dict, sym, tf, row):
    if not _tg_enabled(cfg): return
    _tg_send(cfg, message_closed(sym, tf, row))
