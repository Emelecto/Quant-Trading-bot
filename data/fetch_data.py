"""
Módulo de obtención de datos de mercado (cripto diario).

Responsabilidad única: descargar OHLCV desde el exchange vía ccxt,
guardar crudo en data/raw y limpio en data/processed.
Funciones puras y testeables: no dependen de estado global.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import ccxt


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


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
    path = RAW_DIR / fname
    df.to_csv(path, index=False)
    return path


def load_raw(symbol: str) -> pd.DataFrame:
    fname = symbol.replace("/", "_") + ".csv"
    path = RAW_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"No existe data/raw/{fname}. Ejecuta fetch_and_persist().")
    return pd.read_csv(path, parse_dates=["timestamp"])


def ensure_raw(symbol: str = "BTC/USDT", lookback_years: int = 3) -> pd.DataFrame:
    """Carga datos crudos; si no existen localmente, los descarga de Binance.

    Usado por el dashboard/paper-trader para que funcionen en entornos sin
    datos pre-guardados (ej. Streamlit Cloud). Si la descarga falla, lanza
    un error claro.
    """
    fname = symbol.replace("/", "_") + ".csv"
    path = RAW_DIR / fname
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
    path = PROCESSED_DIR / fname
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
