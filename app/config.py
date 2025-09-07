"""
Global configuration loader for TRADER_AI (local MVP).
Loads policy.yaml and .env (read via os.environ).
All user-facing texts and comments are in English.
"""
from pathlib import Path
import os
import yaml
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
POLICY = yaml.safe_load((ROOT / "policy.yaml").read_text())

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Fees:
    taker_bps: float = float(os.getenv("TAKER_FEE_BPS", POLICY["execution"]["taker_fee_bps_default"]))
    maker_bps: float = float(os.getenv("MAKER_FEE_BPS", POLICY["execution"]["maker_fee_bps_default"]))
    slippage_pct: float = float(os.getenv("AVG_SLIPPAGE_PCT", POLICY["execution"]["avg_slippage_pct"]))

PROFIT_BASIS = os.getenv("PROFIT_BASIS", POLICY["execution"]["profit_basis"])
EXCHANGE = os.getenv("EXCHANGE", "bitget")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTCUSDT").split(",") if s.strip()]
