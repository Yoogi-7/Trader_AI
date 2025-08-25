import sqlite3, yaml
from contextlib import contextmanager

from core.schema import ensure_base_schema, migrate_signals_schema
from core.signals import read_candles, generate_signal, insert_signal
from core.ml import load_meta_model, predict_pwin
from core.regime import compute_regime
from core.notify import send_new_signal

@contextmanager
def db_conn(db_path: str):
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _fees_total_pct(cfg: dict) -> float:
    fees_cfg = cfg["signals"]["fees"]
    mode = cfg["models"]["meta"]["gating"].get("costs_mode", "auto")
    if mode == "maker":
        f = float(fees_cfg["maker_pct"])
    elif mode == "taker":
        f = float(fees_cfg["taker_pct"])
    else:
        f = float(fees_cfg["maker_pct"]) if fees_cfg.get("assume_maker", False) else float(fees_cfg["taker_pct"])
    return 2.0 * f

def _net_move_pct(direction: str, entry: float, level: float) -> float:
    return (level / entry - 1.0) * 100.0 if direction == "long" else (entry / level - 1.0) * 100.0

def _ev_pct(sig: dict, cfg: dict, p_win: float) -> float:
    fees_total = _fees_total_pct(cfg)
    slip = float(cfg["signals"]["slippage_pct"])
    g_win = _net_move_pct(sig["direction"], float(sig["entry"]), float(sig["tp1"]))
    g_loss = _net_move_pct(sig["direction"], float(sig["entry"]), float(sig["sl"]))
    g_win_net = g_win - fees_total - slip
    g_loss_net = g_loss - fees_total - slip
    return p_win * g_win_net + (1.0 - p_win) * g_loss_net

def _resolve_thresholds(cfg: dict, symbol: str, tf: str, regime: dict):
    gcfg = cfg["models"]["meta"]["gating"]
    use_ev = bool(gcfg.get("use_ev", False))
    min_ev = float(gcfg.get("min_ev_pct", 0.0))
    thr = float(cfg["models"]["meta"].get("threshold", 0.6))

    # 1) per_pair_tf override
    per = gcfg.get("per_pair_tf", {}) or {}
    if symbol in per and tf in per[symbol]:
        o = per[symbol][tf]
        if "min_ev_pct" in o: min_ev = float(o["min_ev_pct"])
        if "threshold"  in o: thr = float(o["threshold"])

    # 2) per_regime override (trend/range + vol_high/low)
    pr = gcfg.get("per_regime", {}) or {}
    if regime:
        if regime.get("trend") == "trend" and "trend" in pr:
            o = pr["trend"]; min_ev = float(o.get("min_ev_pct", min_ev)); thr = float(o.get("threshold", thr))
        if regime.get("trend") == "range" and "range" in pr:
            o = pr["range"]; min_ev = float(o.get("min_ev_pct", min_ev)); thr = float(o.get("threshold", thr))
        vol_key = f"vol_{regime.get('vol','normal')}"
        if vol_key in pr:
            o = pr[vol_key]; min_ev = float(o.get("min_ev_pct", min_ev)); thr = float(o.get("threshold", thr))

    return use_ev, min_ev, thr

def scan_once():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    with db_conn(db_path) as conn:
        ensure_base_schema(conn)
        migrate_signals_schema(conn)

        exch = cfg["exchange"]["id"]
        model, feat_names = (None, None)
        use_meta = bool(cfg.get("models", {}).get("meta", {}).get("enabled", False))
        if use_meta:
            model, feat_names = load_meta_model(cfg["models"]["meta"]["model_path"])

        for sym in cfg["symbols"]:
            for tf in cfg["timeframes"]:
                try:
                    df = read_candles(conn, exch, sym, tf, limit=max(600, cfg["signals"]["lookback_candles"] + 50))
                    sig = generate_signal(df, cfg)
                    if not sig:
                        continue

                    regime = compute_regime(df)
                    status_override = None
                    ev_val = None
                    p = None

                    if use_meta and model is not None and feat_names is not None:
                        p = predict_pwin(df, sig, cfg, model, feat_names)
                        sig["ml_p"] = float(p)
                        sig["ml_model"] = "xgb_v1"

                        use_ev, min_ev, thr = _resolve_thresholds(cfg, sym, tf, regime)
                        if use_ev:
                            ev_val = _ev_pct(sig, cfg, p)
                            if ev_val < min_ev:
                                status_override = "FILTERED"
                                print(f"[FILTER EV] {sym} {tf} {sig['direction']} p={p:.2f} EV={ev_val:.2f}% < {min_ev}% [{regime}]")
                            else:
                                print(f"[PASS  EV] {sym} {tf} {sig['direction']} p={p:.2f} EV={ev_val:.2f}% ≥ {min_ev}% [{regime}]")
                        else:
                            if p < thr:
                                status_override = "FILTERED"
                                print(f"[FILTER p] {sym} {tf} {sig['direction']} p={p:.2f} < {thr} [{regime}]")
                            else:
                                print(f"[PASS  p] {sym} {tf} {sig['direction']} p={p:.2f} ≥ {thr} [{regime}]")

                    insert_signal(conn, exch, sym, tf, sig, status_override=status_override)

                    # powiadom nowy sygnał (tylko te, które przeszły filtr = PENDING)
                    if status_override is None and cfg.get("notify", {}).get("telegram", {}).get("send_on", {}).get("new_signal", True):
                        send_new_signal(cfg, exch, sym, tf, sig, p, ev_val)
                except Exception as e:
                    print(f"[ERR] {sym} {tf}: {e}")

if __name__ == "__main__":
    scan_once()
