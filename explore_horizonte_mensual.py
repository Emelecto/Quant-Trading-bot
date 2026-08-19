"""Exploracion de edge: horizonte MENSUAL (21d) con features alternativas.

Hallazgo (BTC, walk-forward 2022-2025):
  - dir_acc OOS ~57.7% (sobre el azar de 50%)
  - Sharpe anualizado simple ~0.79 (primera vez > 0 fuera de muestra)

Este script VALIDA el edge mensual BAJO LA CAPA DE RIESGO del motor
(SL=2*ATR, TP 1:3, filtro ADX) usando las features mensuales inyectadas
al run_walk_forward via feature_df.

Features NO son tecnicas clasicas (RSI/MACD) sino momentum/vol-state:
  ret_21, ret_63, ret_7, vol_ratio (vol_21/vol_63), dist_ma63.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from models.linear_regression_model import LinearRegressionModel
from models.xgboost_model import XGBoostModel
from models.monte_carlo_model import MonteCarloModel
from backtest.engine import run_walk_forward
from backtest import metrics as M


def monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    feat = pd.DataFrame(index=df.index)
    feat["ret_21"] = close.pct_change(21)
    feat["ret_63"] = close.pct_change(63)
    feat["ret_7"] = close.pct_change(7)
    v21 = close.pct_change().rolling(21).std()
    v63 = close.pct_change().rolling(63).std()
    feat["vol_ratio"] = v21 / v63
    feat["dist_ma63"] = (close - close.rolling(63).mean()) / close.rolling(63).std()
    feat["target"] = ((close.pct_change(21).shift(-21) > 0.5 * v21) &
                      close.pct_change(21).shift(-21).notna()).astype(int)
    return feat.dropna()


def walk_forward_monthly(symbol="BTC/USDT"):
    """OOS dir_acc + Sharpe simple (sin riesgo) de las features mensuales."""
    df = fd.ensure_raw(symbol)
    df = fd.clean_ohlcv(df)
    feat = monthly_features(df)
    X = feat.drop(columns=["target"])
    y = feat["target"]
    close = df["close"]
    fwd = close.pct_change(21).shift(-21).reindex(X.index)

    accs, rets = [], []
    start = 0
    tw, te, st = 252, 63, 63
    n = len(X)
    while start + tw + te <= n:
        tr, tr_y = X.iloc[start:start + tw], y.iloc[start:start + tw]
        teX = X.iloc[start + tw:start + tw + te]
        m = XGBoostModel()
        m.fit(tr, tr_y)
        s = m.predict(teX)
        common = s.index.intersection(fwd.dropna().index)
        sig, r = s.loc[common], fwd.loc[common]
        accs.append((np.sign(sig) == np.sign(r)).mean())
        rets.append(np.sign(sig) * r)
        start += st
    allr = pd.concat(rets).dropna()
    sharpe = np.sqrt(17.4) * allr.mean() / allr.std()  # ~17.4 ventanas 21d/año
    return {"oos_acc": float(np.mean(accs)), "sharpe_ann": float(sharpe), "n": int(len(allr))}


def validate_with_risk(symbol="BTC/USDT"):
    """Edge mensual con la capa de riesgo del motor (SL/TP 1:3, ADX)."""
    df = fd.ensure_raw(symbol)
    df = fd.clean_ohlcv(df)
    feat = monthly_features(df)
    models = [
        XGBoostModel(),
        LinearRegressionModel(),
        MonteCarloModel(n_paths=200, random_state=42),
    ]
    r = run_walk_forward(
        df, models, ensemble_method="accuracy",
        train_window=252, test_window=63, step=63,
        atr_mult=2.0, rr_ratio=3.0, adx_threshold=20.0,
        feature_df=feat,
    )
    bh = df["close"].pct_change().reindex(r.index).dropna()
    r = r.reindex(bh.index)
    eq = (1 + r.fillna(0)).cumprod()
    m = M.all_metrics(r.fillna(0), eq)
    mbh = M.all_metrics(bh, (1 + bh).cumprod())
    return m, mbh


if __name__ == "__main__":
    print("=== Edge mensual SIN riesgo (baseline) ===")
    res = walk_forward_monthly("BTC/USDT")
    print(f"OOS acc: {res['oos_acc']:.4f} | Sharpe ann: {res['sharpe_ann']:.3f} | n={res['n']}")

    print("\n=== Edge mensual CON capa de riesgo (motor) ===")
    m, mbh = validate_with_risk("BTC/USDT")
    print(f"{'Métrica':<14}{'Ensemble(mensual+riesgo)':>24}{'Buy & Hold':>14}")
    for k in ["sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor", "cagr"]:
        print(f"{k:<14}{m[k]:>24.4f}{mbh[k]:>14.4f}")
