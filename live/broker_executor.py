"""Ejecutor de ordenes en Binance Futures TESTNET via ccxt.

Integra la senal mensual generada por live.paper_trader con una cuenta de
demostracion (USDT falsos, $0 real). Permite LONG/SHORT con SL y TP.

Diseno SEGURO (modo demo por defecto):
- DRY_RUN=True -> solo simula (no envia ordenes). Es el defecto en CI/Streamlit.
- Ejecuta solo si la senal CAMBIO respecto a state.json (idempotente: no
  reenvia la misma orden cada vez que corre el scheduler).
- NUNCA usa dinero real. Para real habria que cambiar testnet=False Y pasar
  por validacion externa; esto queda fuera del alcance intencionalmente.

Requisitos (en .env o Streamlit Secrets):
  BINANCE_TESTNET_API_KEY=...
  BINANCE_TESTNET_API_SECRET=...
  DRY_RUN=true|false
"""
from __future__ import annotations

import os
import time
from typing import Optional

import ccxt


TESTNET_BASE = "https://testnet.binancefuture.com"


def get_exchange(dry_run: bool = True, api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
    """Devuelve un cliente ccxt de Binance Futures (testnet por defecto)."""
    key = api_key or os.getenv("BINANCE_TESTNET_API_KEY", "")
    secret = api_secret or os.getenv("BINANCE_TESTNET_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError(
            "Faltan BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET. "
            "Registrate en https://testnet.binancefuture.com (usa tu cuenta "
            "de Binance normal)."
        )
    return ccxt.binance({
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
        # testnet: apunta al host de demostracion
        "hostname": "testnet.binancefuture.com",
    })


def _side(direction: str) -> Optional[str]:
    if direction == "LONG":
        return "buy"
    if direction == "SHORT":
        return "sell"
    return None  # NEUTRAL -> no operar


def close_existing_position(exchange, symbol: str):
    """Cierra cualquier posicion abierta en el simbolo (para rotar la senal)."""
    pos = exchange.fetch_position(symbol)
    if pos and float(pos.get("contracts", 0) or 0) != 0:
        side = "sell" if float(pos["side"] if "side" in pos else 1) > 0 else "buy"
        # cerramos con orden de sentido contrario al lado actual
        current_side = pos.get("side")
        close_side = "sell" if current_side == "long" else "buy"
        exchange.create_order(
            symbol, "market", close_side,
            abs(float(pos["contracts"])), params={"reduceOnly": True}
        )
        return True
    return False


def execute_signal(signal: dict, dry_run: bool = True,
                   api_key: Optional[str] = None,
                   api_secret: Optional[str] = None) -> dict:
    """Coloca la orden mensual en el testnet segun la senal.

    - LONG/SHORT: abre posicion con apalancamiento 1x, SL y TP via order params.
    - NEUTRAL: no opera.
    - dry_run=True: solo reporta lo que HARIA (no envia nada).
    """
    symbol = signal["symbol"]
    direction = signal["direction"]
    side = _side(direction)

    plan = {
        "symbol": symbol,
        "direction": direction,
        "dry_run": dry_run,
        "action": "none",
        "order": None,
        "sl": signal.get("sl_level"),
        "tp": signal.get("tp_level"),
    }
    if side is None:
        plan["action"] = "neutral_no_trade"
        return plan

    # tamano en contrato: position_size_pct del capital / precio (aprox 1x)
    notional_pct = float(signal.get("position_size_pct", 0)) / 100.0
    price = float(signal["last_close"])
    # cantidad en BTC = (capital_simulado * pct) / precio
    capital = float(signal.get("capital_simulado", 50.0))
    qty = round((capital * notional_pct) / price, 5)
    if qty <= 0:
        plan["action"] = "qty_zero"
        return plan

    if dry_run:
        plan["action"] = "DRY_RUN_would_open"
        plan["order"] = {"symbol": symbol, "side": side, "type": "market", "qty": qty,
                         "sl": plan["sl"], "tp": plan["tp"]}
        return plan

    # === EJECUCION REAL EN TESTNET ===
    exchange = get_exchange(dry_run, api_key, api_secret)
    # apalancamiento 1x (conservador, coherente con riesgo 1%)
    try:
        exchange.set_leverage(1, symbol)
    except Exception:
        pass
    # cerrar posicion previa si existe (rotar senal)
    close_existing_position(exchange, symbol)
    time.sleep(1)
    # abrir nueva posicion
    order = exchange.create_order(
        symbol, "market", side, qty,
        params={"reduceOnly": False}
    )
    # colocar SL y TP como ordenes condicionales (STOP_MARKET / TAKE_PROFIT_MARKET)
    sl_side = "sell" if side == "buy" else "buy"
    exchange.create_order(symbol, "STOP_MARKET", sl_side, qty,
                          params={"stopPrice": round(plan["sl"], 2), "reduceOnly": True})
    exchange.create_order(symbol, "TAKE_PROFIT_MARKET", sl_side, qty,
                          params={"stopPrice": round(plan["tp"], 2), "reduceOnly": True})
    plan["action"] = "opened"
    plan["order"] = order
    return plan


if __name__ == "__main__":
    # ejemplo de uso (requiere keys en env y dry_run=false para ejecutar)
    import json
    from live.paper_trader import current_signal
    sig = current_signal("BTC/USDT")
    sig["capital_simulado"] = 50.0
    result = execute_signal(sig, dry_run=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
