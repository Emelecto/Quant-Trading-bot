"""
Construcción del feature matrix a partir de OHLCV limpio.

Produce features técnicas + retornos/volatilidad, más la variable objetivo
(target): 1 si el retorno futuro supera un umbral de volatilidad, si no 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import indicators as ind


def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int = 1,
    vol_threshold_mult: float = 0.5,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    out["sma_20"] = ind.sma(close, 20)
    out["ema_20"] = ind.ema(close, 20)
    out["rsi_14"] = ind.rsi(close, 14)
    macd_line, macd_sig, macd_hist = ind.macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    upper, lower, pct_b = ind.bollinger(close, 20, 2.0)
    out["boll_pct_b"] = pct_b
    out["atr_14"] = ind.atr(df, 14)
    out["adx_14"] = ind.adx(df, 14)

    out["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = ind.rolling_returns(close, 5)
    out["vol_20"] = ind.rolling_volatility(close, 20)

    # target usa info FUTURA (retorno a horizon barra(s)) -> sin look-ahead porque es la salida
    future_ret = close.pct_change(horizon).shift(-horizon)
    vol = out["vol_20"]
    threshold = vol_threshold_mult * vol
    out["target"] = ((future_ret > threshold) & (future_ret.notna())).astype(int)

    # descartar columnas no numéricas (ej. timestamp) que puedan llegar en el df
    out = out.select_dtypes(include=[np.number])
    if "target" not in out.columns:
        out["target"] = ((future_ret > threshold) & (future_ret.notna())).astype(int)

    out = out.dropna().iloc[:-horizon]
    return out


if __name__ == "__main__":
    from data import fetch_data as fd
    for sym in ["BTC/USDT", "ETH/USDT"]:
        df = fd.load_raw(sym)
        df = fd.clean_ohlcv(df)
        feats = build_feature_matrix(df)
        print(f"[{sym}] features shape: {feats.shape}, win rate objetivo: {feats['target'].mean():.3f}")
