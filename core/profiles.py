# core/profiles.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SignalProfile:
    name: str
    # Reżim rynku
    bullish_only: bool
    min_atr_pct: float
    # Walidacja
    n_splits: int
    embargo: int
    # Etykietowanie (triple barrier)
    tb_horizon: int
    tb_atr_period: int
    tb_tp_mult: float
    tb_sl_mult: float
    # Celowanie w winrate
    target_precision: float
    min_signals: int
    # Meta-labeling
    enable_meta: bool

PROFILES = {
    "Zachowawcze": SignalProfile(
        name="Zachowawcze",
        bullish_only=True,
        min_atr_pct=1.0,      # tylko wyraźniejsza zmienność
        n_splits=6,
        embargo=50,
        tb_horizon=80,
        tb_atr_period=14,
        tb_tp_mult=2.0,
        tb_sl_mult=1.0,
        target_precision=0.84, # wysoki winrate
        min_signals=30,
        enable_meta=True,      # filtr jakości (meta) włączony
    ),
    "Wyśrodkowane": SignalProfile(
        name="Wyśrodkowane",
        bullish_only=True,
        min_atr_pct=0.5,
        n_splits=5,
        embargo=30,
        tb_horizon=60,
        tb_atr_period=14,
        tb_tp_mult=2.0,
        tb_sl_mult=1.0,
        target_precision=0.72,
        min_signals=80,
        enable_meta=True,
    ),
    "Agresywne": SignalProfile(
        name="Agresywne",
        bullish_only=False,     # bez wymogu trendu
        min_atr_pct=0.2,
        n_splits=4,
        embargo=10,
        tb_horizon=40,
        tb_atr_period=14,
        tb_tp_mult=2.0,
        tb_sl_mult=1.0,
        target_precision=0.60,  # niższa precyzja → więcej sygnałów
        min_signals=150,
        enable_meta=False,      # więcej recallu
    ),
}

def get_profile(name: str) -> SignalProfile:
    return PROFILES.get(name, PROFILES["Wyśrodkowane"])
