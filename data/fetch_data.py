"""Módulo de obtención de datos de mercado (cripto diario).

Responsabilidad única: descargar OHLCV desde el exchange vía ccxt,
guardar crudo en cache y limpio en procesado.

Cloud-safe: NO crea directorios ni escribe dentro del repo al importar
(el filesystem de Streamlit Cloud es de solo lectura). El cache de datos
usa /tmp en la nube para no romper el import ni la escritura.
"""
from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import ccxt


def _data_dir(sub: str) -> Path:
    """Directorio de datos escribible.

    En Streamlit Cloud el repo es de solo lectura, asi que usamos /tmp.
    Localmente (donde el repo es escribible) usamos data/ dentro del repo.
    """
    if os.access(str(Path(__file__).resolve().parent.parent), os.W_OK):
        d = Path(__file__).resolve().parent.parent / "data" / sub
    else:
        d = Path(tempfile.gettempdir()) / "deepfin_data" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    lookback_years: int = 3,
    exchange_name: str = "binance",
    rate_limit_sleep: float = 0.25,
) -> pd.DataFrame:
    """Descarga velas OHLCV desde el exchange vía ccxt."""
    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    since_ms = int((pd.Timestamp.utcnow() - pd.DateOffset(years=lookback_years)).timestamp() * 1000)
    all_rows: list[list] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + 1
        time.sleep(rate_limit_sleep)
        if len(all_rows) and all_rows[-1][0] >= pd.Timestamp.utcnow().timestamp() * 1000:
            break
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def save_raw(df: pd.DataFrame, symbol: str) -> Path:
    fname = symbol.replace("/", "_") + ".csv"
    path = _data_dir("raw") / fname
    df.to_csv(path, index=False)
    return path


def load_raw(symbol: str) -> pd.DataFrame:
    fname = symbol.replace("/", "_") + ".csv"
    path = _data_dir("raw") / fname
    if not path.exists():
        raise FileNotFoundError(f"No existe cache para {symbol}. Ejecuta ensure_raw().")
    return pd.read_csv(path, parse_dates=["timestamp"])


def ensure_raw(symbol: str = "BTC/USDT", lookback_years: int = 3) -> pd.DataFrame:
    """Carga datos; si no existen localmente en cache, los descarga de Binance.

    Cloud-safe: usa /tmp como cache cuando el repo es de solo lectura.
    """
    fname = symbol.replace("/", "_") + ".csv"
    path = _data_dir("raw") / fname
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    raw = fetch_ohlcv(symbol, lookback_years=lookback_years)
    save_raw(raw, symbol)
    return raw


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    return df


def persist_clean(df: pd.DataFrame, symbol: str) -> Path:
    fname = symbol.replace("/", "_") + "_clean.csv"
    path = _data_dir("processed") / fname
    df.to_csv(path, index=False)
    return path


def fetch_and_persist(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    lookback_years: int = 3,
    exchange_name: str = "binance",
) -> pd.DataFrame:
    raw = fetch_ohlcv(symbol, timeframe, lookback_years, exchange_name)
    save_raw(raw, symbol)
    clean = clean_ohlcv(raw)
    persist_clean(clean, symbol)
    print(f"[fetch] {symbol}: {len(clean)} velas limpias guardadas.")
    return clean


if __name__ == "__main__":
    for sym in ["BTC/USDT", "ETH/USDT"]:
        fetch_and_persist(symbol=sym, lookback_years=3)
