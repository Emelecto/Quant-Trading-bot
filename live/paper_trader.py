"""
Paper trading en vivo (simulado) — estrategia mensual BTC con riesgo 1%.

Genera la senal mensual actual con los datos mas recientes de Binance,
calcula la posicion sugerida con tamano de posicion por riesgo (1% del
capital simulado por trade), y persiste el estado en state.json.

NO ejecuta ordenes reales: es paper trading (capital simulado, por defecto $50).
Para operar con dinero real se requeriria API key del exchange + validacion.

Uso:
  python -m live.paper_trader          # genera senal y actualiza state.json
  python -m live.paper_trader --capital 50 --risk 0.01
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from explore_horizonte_mensual import monthly_features
from models.xgboost_model import XGBoostModel

STATE_PATH = Path(__file__).resolve().parent / "state.json"
HOLD = 21          # dias de mantenimiento
ATR_MULT = 2.0     # SL = 2 * vol_21d
RR = 3.0           # TP = 1:3 respecto al SL
DEFAULT_CAPITAL = 50.0
DEFAULT_RISK = 0.01


def current_signal(symbol="BTC/USDT") -> dict:
    """Entrena con todo el historico y predice la senal del ultimo dia util.

    Nota honesta: esto NO es walk-forward (usa todos los datos para entrenar).
    Es la senal 'en vivo' que el bot seguira mes a mes. El edge OOS ya fue
    validado por separado; aqui solo se produce la senal operativa.
    """
    df = fd.load_raw(symbol)
    df = fd.clean_ohlcv(df)
    feat = monthly_features(df)
    X = feat.drop(columns=["target"])
    y = feat["target"]
    model = XGBoostModel()
    model.fit(X, y)
    last_signal = float(model.predict(X.iloc[[-1]]).iloc[0])
    close = df["close"]
    vol21 = float(close.pct_change().rolling(21).std().iloc[-1])
    entry = float(close.iloc[-1])
    sl_level = entry * (1 - ATR_MULT * vol21)
    tp_level = entry * (1 + RR * ATR_MULT * vol21)
    sl_frac = abs(entry - sl_level) / entry
    position_size = min(DEFAULT_RISK / sl_frac, 1.0) if sl_frac > 0 else 0.0
    direction = int(np.sign(last_signal))
    return {
        "symbol": symbol,
        "date": str(pd.Timestamp.now().date()),
        "last_close": entry,
        "signal_score": round(last_signal, 4),
        "direction": "LONG" if direction > 0 else ("SHORT" if direction < 0 else "NEUTRAL"),
        "vol_21d": round(vol21, 5),
        "sl_level": round(sl_level, 2),
        "tp_level": round(tp_level, 2),
        "sl_frac": round(sl_frac, 5),
        "position_size_pct": round(position_size * 100, 2),
        "risk_per_trade_pct": DEFAULT_RISK * 100,
        "hold_days": HOLD,
    }


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def run(capital: float = DEFAULT_CAPITAL, risk: float = DEFAULT_RISK):
    sig = current_signal("BTC/USDT")
    sig["capital_simulado"] = capital
    sig["risk_per_trade"] = risk
    state = load_state()
    # conservar historial de senales
    history = state.get("history", [])
    history.append({k: sig[k] for k in ["date", "signal_score", "direction", "last_close"]})
    sig["history"] = history[-50:]
    save_state(sig)
    return sig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--risk", type=float, default=DEFAULT_RISK)
    args = ap.parse_args()
    res = run(args.capital, args.risk)
    print(json.dumps(res, indent=2, ensure_ascii=False))
