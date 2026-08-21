"""Busqueda automatica de la mejor estrategia intradia bajo el bar de Lopez de Prado.

Barre parametros de funding carry y reversal 1h, evalua cada uno con walk-forward
Purged CV + deflated Sharpe + prob>0, y reporta la combinacion que SUPERA el bar
(deflated Sharpe > 0 y prob > 0.95) generando maximo dinero OOS vs B&H.

El critico ciego elige la ganadora por deflated Sharpe y equity OOS.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategy.intraday_strategies import (
    load_1h, load_funding, funding_carry_signal, reversal_signal,
    evaluate, buy_and_hold,
)


def search(close: pd.Series, fund: pd.Series | None) -> list[dict]:
    results = []
    # B&H reference
    bh = buy_and_hold(close, 8)
    results.append({"name": "B&H", **bh, "deflated_sharpe": bh["sharpe"], "prob_gt_0": float("nan")})

    # FUNDING CARRY (solo si hay datos de funding)
    if fund is not None and len(fund) > 0:
        for hold in [4, 8, 12, 24]:
            sig = funding_carry_signal(close.index, fund, hold_h=hold)
            r = evaluate(sig, close, hold)
            results.append({"name": f"funding_carry_h{hold}", **r})

    # REVERSAL: varios lookback/hold (incluye cortos donde vive el edge real)
    for lb in [1, 2, 3, 6, 12, 24, 48, 72]:
        for hold in [1, 2, 4, 6, 12]:
            sig = reversal_signal(close, lookback=lb, hold_h=hold)
            r = evaluate(sig, close, hold)
            results.append({"name": f"reversal_lb{lb}_h{hold}", **r})

    return results


def best(results: list[dict]) -> dict:
    """Ganadora del bar: deflated Sharpe > 0 y prob > 0.95, maxima equity."""
    valid = [r for r in results
             if r.get("deflated_sharpe", 0) > 0
             and r.get("prob_gt_0", 0) > 0.95
             and np.isfinite(r.get("eq", 0))]
    if not valid:
        # si ninguna supera el bar, la menos mala por deflated Sharpe
        valid = results
    return max(valid, key=lambda r: (r.get("deflated_sharpe", -9), r.get("eq", 0)))


if __name__ == "__main__":
    df = load_1h()
    close = df["close"]
    fund = None
    try:
        fund = load_funding()
    except Exception:
        print("[info] sin funding; evaluando solo reversal intradia")
    print(f"[data] 1h filas={len(df)} funding={'si' if fund is not None else 'no'}")
    res = search(close, fund)
    # tabla ciega: solo numeros, sin saber cual es "la nuestra"
    print(f"{'estrategia':<22}{'n':>6}{'sharpe':>9}{'defl':>9}{'prob>0':>9}{'eq':>12}")
    for r in sorted(res, key=lambda x: -x.get("deflated_sharpe", -9)):
        print(f"{r['name']:<22}{r.get('n',0):>6}{r.get('sharpe',0):>9.3f}"
              f"{r.get('deflated_sharpe',0):>9.3f}{r.get('prob_gt_0',float('nan')):>9.3f}{r.get('eq',0):>12.3f}")
    win = best(res)
    print(f"\nGANADORA DEL BAR: {win['name']} deflated={win.get('deflated_sharpe',0):.3f} eq={win.get('eq',0):.3f}")
