"""Paper broker local para DeepFin (estilo backtrader.BackBroker).

Mantiene una posicion MENSUAL de BTC con marca a mercado diaria usando el
precio REAL de Binance (ccxt, spot, gratis, sin cuenta). Registra cada trade
en un ledger persistente (live/trades.json) con entry/exit/size/PNL/SL/TP y
el motivo de cierre. Es la pieza A del gauntlet loop: replicar la disciplina
de backtrader (cash virtual, mark-to-market, trades persistentes) sin broker.

Flujo:
  - Al inicio del mes: si la senal cambio, abre posicion (LONG/SHORT) al cierre
    actual con SL/TP de la senal y riesgo 1%.
  - Cada dia (mark_to_market): actualiza PNL no realizado vs precio real.
  - Cierre: si el precio toca SL o TP, o pasan 21 dias -> liquida y registra.

Uso:
  python -m live.ledger_bot --action open     # abre segun senal del mes
  python -m live.ledger_bot --action mark     # marca a mercado (correr diario)
  python -m live.ledger_bot --action status   # estado actual
"""
from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from live import paper_trader as pt


LEDGER_PATH = Path(__file__).resolve().parent / "trades.json"
HOLD = 21
ATR_MULT = 2.0
RR = 3.0
DEFAULT_CAPITAL = 50.0
DEFAULT_RISK = 0.01


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return {"capital": DEFAULT_CAPITAL, "trades": [], "open_position": None}


def _save_ledger(ledger: dict):
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def _real_price(symbol: str = "BTC/USDT") -> float:
    """Precio real de Binance (spot) via ccxt. No requiere cuenta."""
    import ccxt
    ex = ccxt.binance({"enableRateLimit": True})
    ticker = ex.fetch_ticker(symbol)
    return float(ticker["last"])


def open_position(capital: float = DEFAULT_CAPITAL, risk: float = DEFAULT_RISK) -> dict:
    """Abre posicion segun la senal mensual actual."""
    ledger = _load_ledger()
    if ledger.get("open_position"):
        return {"status": "already_open", "position": ledger["open_position"]}

    sig = pt.current_signal("BTC/USDT")
    sig["capital_simulado"] = capital
    sig["risk_per_trade"] = risk

    if sig["direction"] == "NEUTRAL" or sig["position_size_pct"] <= 0:
        return {"status": "neutral_no_open", "signal": sig}

    entry = _real_price("BTC/USDT")
    side = "long" if sig["direction"] == "LONG" else "short"
    # tamano en BTC: capital * pct / entry
    qty = (capital * (sig["position_size_pct"] / 100.0)) / entry
    position = {
        "symbol": "BTC/USDT",
        "side": side,
        "entry_price": entry,
        "qty": round(qty, 6),
        "sl": sig["sl_level"],
        "tp": sig["tp_level"],
        "open_date": str(datetime.utcnow().date()),
        "close_date": None,
        "exit_price": None,
        "pnl": 0.0,
        "pnl_pct": 0.0,
        "exit_reason": None,
        "signal_score": sig["signal_score"],
        "unrealized_pnl": 0.0,
    }
    ledger["open_position"] = position
    _save_ledger(ledger)
    return {"status": "opened", "position": position}


def mark_to_market() -> dict:
    """Marca a mercado la posicion abierta; cierra si toca SL/TP o vence 21d."""
    ledger = _load_ledger()
    pos = ledger.get("open_position")
    if not pos:
        return {"status": "no_open_position"}

    price = _real_price("BTC/USDT")
    if pos["side"] == "long":
        unreal = (price - pos["entry_price"]) * pos["qty"]
    else:
        unreal = (pos["entry_price"] - price) * pos["qty"]
    pos["unrealized_pnl"] = round(unreal, 2)

    # chequeo de cierre
    hit_sl = (pos["side"] == "long" and price <= pos["sl"]) or \
             (pos["side"] == "short" and price >= pos["sl"])
    hit_tp = (pos["side"] == "long" and price >= pos["tp"]) or \
             (pos["side"] == "short" and price <= pos["tp"])
    days_open = (datetime.utcnow() - datetime.strptime(pos["open_date"], "%Y-%m-%d")).days
    expired = days_open >= HOLD

    if hit_sl:
        return _close(ledger, pos, pos["sl"], "stop_loss")
    if hit_tp:
        return _close(ledger, pos, pos["tp"], "take_profit")
    if expired:
        return _close(ledger, pos, price, "hold_period_end")

    _save_ledger(ledger)
    return {"status": "marked", "position": pos, "price": price}


def _close(ledger: dict, pos: dict, exit_price: float, reason: str) -> dict:
    if pos["side"] == "long":
        pnl = (exit_price - pos["entry_price"]) * pos["qty"]
    else:
        pnl = (pos["entry_price"] - exit_price) * pos["qty"]
    pos["exit_price"] = round(exit_price, 2)
    pos["exit_reason"] = reason
    pos["close_date"] = str(datetime.utcnow().date())
    pos["pnl"] = round(pnl, 2)
    pos["pnl_pct"] = round(pnl / ledger["capital"] * 100, 2)
    ledger["capital"] = round(ledger["capital"] + pnl, 2)
    ledger["trades"].append(pos)
    ledger["open_position"] = None
    _save_ledger(ledger)
    return {"status": "closed", "trade": pos, "capital_after": ledger["capital"]}


def status() -> dict:
    ledger = _load_ledger()
    return {"capital": ledger["capital"], "open_position": ledger["open_position"],
            "n_trades": len(ledger["trades"]),
            "trades": ledger["trades"][-10:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["open", "mark", "status"], default="status")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--risk", type=float, default=DEFAULT_RISK)
    args = ap.parse_args()
    if args.action == "open":
        out = open_position(args.capital, args.risk)
    elif args.action == "mark":
        out = mark_to_market()
    else:
        out = status()
    print(json.dumps(out, indent=2, ensure_ascii=False))
