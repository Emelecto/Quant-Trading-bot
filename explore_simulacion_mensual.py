"""Simulacion mensual correcta del edge: posicion 21d + SL/TP de volatilidad 21d.

El motor run_walk_forward simula barra a barra (diario) y anula senales
mensuales. Este script simula el horizonte MENSUAL real:
  - se entra segun la senal mensual (long/short)
  - se mantiene 21 dias (o hasta SL/TP de vol 21d)
  - SL = 2*vol_21d del entry, TP = 1:3 respecto al SL
Luego compara vs Buy & Hold (mismo horizonte de senales).

Incluye tamaño de posicion por riesgo (risk_per_trade): en lugar de apostar
el 100% del capital, se apuesta solo la fraccion que hace que un SL cueste
exactamente risk_per_trade del capital -> reduce drasticamente el drawdown.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from models.xgboost_model import XGBoostModel
from backtest import metrics as M
from explore_horizonte_mensual import monthly_features


def simulate_monthly_with_risk(symbol="BTC/USDT", hold=21, atr_mult=2.0, rr=3.0,
                               risk_per_trade=1.0):
    """Simula el edge mensual con riesgo.

    risk_per_trade: fraccion del capital arriesgada por operacion.
      - 1.0  = apuesta 100% del capital (baseline, DD alto)
      - 0.01 = arriesga 1% del capital por trade (SL define el tamaño)
    """
    df = fd.load_raw(symbol)
    df = fd.clean_ohlcv(df)
    feat = monthly_features(df)
    X = feat.drop(columns=["target"])
    y = feat["target"]
    close = df["close"]

    # walk-forward para generar senales en test (fuera de muestra)
    signals = pd.Series(index=X.index, dtype=float)
    start = 0
    tw, te, st = 252, 63, 63
    n = len(X)
    while start + tw + te <= n:
        tr, tr_y = X.iloc[start:start + tw], y.iloc[start:start + tw]
        teX = X.iloc[start + tw:start + tw + te]
        m = XGBoostModel()
        m.fit(tr, tr_y)
        s = m.predict(teX)
        signals.loc[s.index] = s
        start += st
    signals = signals.dropna()

    # volatilidad 21d para SL/TP
    vol21 = close.pct_change().rolling(21).std().reindex(signals.index).ffill()

    # simular con capital variable y tamaño de posicion por riesgo
    capital = 1.0
    eq_curve = [capital]
    eq_dates = [close.index[0]]
    rets = []
    dates = []
    idx = list(signals.index)
    i = 0
    while i < len(idx):
        t = idx[i]
        pos = int(np.sign(signals.loc[t]))
        if pos == 0:
            i += 1
            continue
        entry_i = close.index.get_loc(t)
        end_i = min(entry_i + hold, len(close) - 1)
        entry = close.iloc[entry_i]
        sl_level = entry * (1 - atr_mult * vol21.loc[t])
        tp_level = entry * (1 + rr * atr_mult * vol21.loc[t])
        sl_frac = abs(entry - sl_level) / entry
        if sl_frac <= 0:
            i += 1
            continue
        position_size = risk_per_trade / sl_frac
        position_size = min(position_size, 1.0)
        exited = False
        for j in range(entry_i + 1, end_i + 1):
            price = close.iloc[j]
            if pos > 0 and price <= sl_level:
                tr = (price / entry - 1) * position_size
                capital *= (1 + tr); rets.append(tr); dates.append(close.index[j]); exited = True; break
            if pos > 0 and price >= tp_level:
                tr = (price / entry - 1) * position_size
                capital *= (1 + tr); rets.append(tr); dates.append(close.index[j]); exited = True; break
            if pos < 0 and price >= sl_level:
                tr = -(price / entry - 1) * position_size
                capital *= (1 + tr); rets.append(tr); dates.append(close.index[j]); exited = True; break
            if pos < 0 and price <= tp_level:
                tr = -(price / entry - 1) * position_size
                capital *= (1 + tr); rets.append(tr); dates.append(close.index[j]); exited = True; break
        if not exited:
            exit_price = close.iloc[end_i]
            tr = (exit_price / entry - 1) * pos * position_size
            capital *= (1 + tr); rets.append(tr); dates.append(close.index[end_i])
        eq_curve.append(capital); eq_dates.append(dates[-1])
        i += 1

    r = pd.Series(rets, index=dates).sort_index()
    eq = pd.Series(eq_curve, index=eq_dates).sort_index()
    mres = M.all_metrics(r.fillna(0), eq)
    # benchmark buy & hold sobre el mismo horizonte de senales
    bh_rets = []
    bh_dates = []
    bi = 0
    while bi < len(idx):
        t = idx[bi]
        ei = close.index.get_loc(t)
        ej = min(ei + hold, len(close) - 1)
        bh_rets.append(close.iloc[ej] / close.iloc[ei] - 1)
        bh_dates.append(close.index[ej])
        bi += 1
    rbh = pd.Series(bh_rets, index=bh_dates).sort_index()
    mbh = M.all_metrics(rbh.fillna(0), (1 + rbh.fillna(0)).cumprod())
    return mres, mbh


if __name__ == "__main__":
    for sym in ["BTC/USDT", "ETH/USDT"]:
        for rpt in [1.0, 0.10, 0.01]:
            m, mbh = simulate_monthly_with_risk(sym, risk_per_trade=rpt)
            print(f"[{sym}] risk/trade={rpt:.2f} | Sharpe {m['sharpe']:.3f} MaxDD {m['max_drawdown']:.3f} "
                  f"CAGR {m['cagr']:.3f} | B&H Sharpe {mbh['sharpe']:.3f}")
