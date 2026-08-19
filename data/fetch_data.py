"""Módulo de obtención de datos de mercado (cripto diario).

Responsabilidad única: cargar OHLCV para el backtest/dashboard.

Diseño cloud-safe y robusto (crítico para Streamlit Cloud):
- ccxt se importa DE FORMA DIFERIDA (dentro de fetch_ohlcv). Asi el modulo
  carga aunque ccxt NO este instalado en la nube (Python 3.14) -> evita el
  AttributeError 'has no attribute ensure_raw'.
- Los datos vienen de data/datasets/ (tracked en el repo) -> el dashboard
  funciona en la nube SIN ccxt ni descarga de red.
- NO crea directorios ni escribe en el repo al importar (FS de solo lectura).
"""
from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path

import pandas as pd


# Carpeta de datos tracked en el repo (siempre disponible, incluida en git).
DATASETS_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"


def _writable_cache(sub: str) -> Path:
    """Cache escribible (/tmp en la nube, data/ local si escribible)."""
    if os.access(str(Path(__file__).resolve().parent.parent), os.W_OK):
        d = Path(__file__).resolve().parent.parent / "data" / sub
    else:
        d = Path(tempfile.gettempdir()) / "deepfin_data" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_raw(symbol: str) -> pd.DataFrame:
    """Carga desde data/datasets/ (tracked) o desde cache escribible."""
    fname = symbol.replace("/", "_") + ".csv"
    ds = DATASETS_DIR / fname
    if ds.exists():
        return _read_csv(ds)
    cache = _writable_cache("raw") / fname
    if cache.exists():
        return _read_csv(cache)
    raise FileNotFoundError(f"No hay datos para {symbol}. Ejecuta fetch_and_persist().")


def ensure_raw(symbol: str = "BTC/USDT", lookback_years: int = 3) -> pd.DataFrame:
    """Carga datos de data/datasets/ (优先). Si faltan, intenta descargar (ccxt)."""
    fname = symbol.replace("/", "_") + ".csv"
    ds = DATASETS_DIR / fname
    if ds.exists():
        return _read_csv(ds)
    # fallback: descarga via ccxt (requiere red + ccxt instalado)
    raw = fetch_ohlcv(symbol, lookback_years=lookback_years)
    _writable_cache("raw")
    raw.to_csv(_writable_cache("raw") / fname, index=False)
    return raw


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    lookback_years: int = 3,
    exchange_name: str = "binance",
    rate_limit_sleep: float = 0.25,
) -> pd.DataFrame:
    """Descarga velas OHLCV desde el exchange via ccxt (import diferido)."""
    import ccxt  # diferido: el modulo no falla si ccxt no esta instalado
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
    path = _writable_cache("processed") / fname
    df.to_csv(path, index=False)
    return path


def fetch_and_persist(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    lookback_years: int = 3,
    exchange_name: str = "binance",
) -> pd.DataFrame:
    raw = fetch_ohlcv(symbol, timeframe, lookback_years, exchange_name)
    clean = clean_ohlcv(raw)
    persist_clean(clean, symbol)
    print(f"[fetch] {symbol}: {len(clean)} velas limpias guardadas.")
    return clean


if __name__ == "__main__":
    for sym in ["BTC/USDT", "ETH/USDT"]:
        fetch_and_persist(symbol=sym, lookback_years=3)
