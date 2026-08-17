"""Exploracion de edge: horizonte MENSUAL (21d) con features alternativas.

Hallazgo (BTC, walk-forward 2022-2025):
  - dir_acc OOS ~57.7% (sobre el azar de 50%)
  - Sharpe anualizado de la estrategia ~0.79 (primera vez > 0 fuera de muestra)

Features NO son tecnicas clasicas (RSI/MACD) sino momentum/vol-state:
  ret_21, ret_63, ret_7, vol_ratio (vol_21/vol_63), dist_ma63.
Target: retorno a 21d supera 0.5*vol_21.

Esto sugiere que el edge OOS existe en HORIZONTE LARGO, no diario.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from models.xgboost_model import XGBoostModel


def monthly_features(df: pd.DataFrame):
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
    df = fd.load_raw(symbol)
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
    sharpe = np.sqrt(17.4) * allr.mean() / allr.std()
    return {"oos_acc": float(np.mean(accs)), "sharpe_ann": float(sharpe), "n": int(len(allr))}


if __name__ == "__main__":
    res = walk_forward_monthly("BTC/USDT")
    print(f"OOS acc: {res['oos_acc']:.4f} | Sharpe ann: {res['sharpe_ann']:.3f} | n={res['n']}")
