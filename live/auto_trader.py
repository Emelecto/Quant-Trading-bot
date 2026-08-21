"""Orquestador del bot mensual automatico (demo, Binance Futures Testnet).

Flujo una vez al mes:
  1. Genera la senal mensual (live.paper_trader.current_signal).
  2. Persiste estado en state.json (historial de senales).
  3. Si la senal CAMBIO vs la ultima vez -> ejecuta la orden en el testnet
     (solo si DRY_RUN=false). Si no cambio, no reenvia (idempotente).

Modo seguro por defecto: DRY_RUN=true (no envia ordenes reales en demo).
Para ejecutar de verdad en demo: DRY_RUN=false + keys de testnet configuradas.

Uso:
  python -m live.auto_trader                 # DRY_RUN segun env (defecto true)
  python -m live.auto_trader --execute       # fuerza ejecucion (aun requiere keys)
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live import paper_trader as pt
from live import broker_executor as be


def run(execute: bool = False, capital: float = 50.0, risk: float = 0.01) -> dict:
    dry_run = not execute and os.getenv("DRY_RUN", "true").lower() != "false"

    # 1) senal mensual
    sig = pt.current_signal("BTC/USDT")
    sig["capital_simulado"] = capital
    sig["risk_per_trade"] = risk

    # 2) estado previo
    state = pt.load_state()
    prev_dir = state.get("direction")
    prev_date = state.get("date")

    # 3) idempotencia: solo actuar si la senal cambio o es el primer run
    changed = (prev_dir != sig["direction"]) or (prev_date != sig["date"])

    # 4) ejecutar (o simular)
    if changed:
        result = be.execute_signal(sig, dry_run=dry_run)
        sig["execution"] = result
        sig["executed"] = (not dry_run)
    else:
        sig["execution"] = {"action": "no_change_skipped", "direction": sig["direction"]}
        sig["executed"] = False

    # 5) persistir
    history = state.get("history", [])
    history.append({k: sig[k] for k in ["date", "signal_score", "direction", "last_close"]})
    sig["history"] = history[-50:]
    sig["dry_run"] = dry_run
    pt.save_state(sig)
    return sig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="ejecuta en testnet (requiere keys)")
    ap.add_argument("--capital", type=float, default=50.0)
    ap.add_argument("--risk", type=float, default=0.01)
    args = ap.parse_args()
    res = run(execute=args.execute, capital=args.capital, risk=args.risk)
    print(json.dumps(res, indent=2, ensure_ascii=False))
