"""Simulacion mensual correcta del edge: posicion 21d + SL/TP de volatilidad 21d.

El motor run_walk_forward simula barra a barra (diario) y anula senales
mensuales. Este script simula el horizonte MENSUAL real:
  - se entra segun la senal mensual (long/short)
  - se mantiene 21 dias (o hasta SL/TP de vol 21d)
  - SL = 2*vol_21d del entry, TP = 1:3 respecto al SL
Luego compara vs Buy & Hold.
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


def simulate_monthly_with_risk(symbol="BTC/USDT", hold=21, atr_mult=2.0, rr=3.0):
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

    # simular: en cada senal mensual, mantener 'hold' dias con SL/TP de vol21
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
        # chequear SL/TP dia a dia
        exited = False
        for j in range(entry_i + 1, end_i + 1):
            price = close.iloc[j]
            if pos > 0 and price <= sl_level:
                rets.append(price / entry - 1); dates.append(close.index[j]); exited = True; break
            if pos > 0 and price >= tp_level:
                rets.append(price / entry - 1); dates.append(close.index[j]); exited = True; break
            if pos < 0 and price >= sl_level:
                rets.append(-(price / entry - 1)); dates.append(close.index[j]); exited = True; break
            if pos < 0 and price <= tp_level:
                rets.append(-(price / entry - 1)); dates.append(close.index[j]); exited = True; break
        if not exited:
            exit_price = close.iloc[end_i]
            rets.append((exit_price / entry - 1) * pos)
            dates.append(close.index[end_i])
        i += 1  # siguiente senal mensual

    r = pd.Series(rets, index=dates).sort_index()
    eq = (1 + r.fillna(0)).cumprod()
    mres = M.all_metrics(r.fillna(0), eq)
    bh = df["close"].pct_change().reindex(r.index).dropna()
    mbh = M.all_metrics(bh, (1 + bh).cumprod())
    return mres, mbh


if __name__ == "__main__":
    m, mbh = simulate_monthly_with_risk("BTC/USDT")
    print(f"{'Métrica':<14}{'Mensual+riesgo(21d)':>22}{'Buy & Hold':>14}")
    for k in ["sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor", "cagr"]:
        print(f"{k:<14}{m[k]:>22.4f}{mbh[k]:>14.4f}")
