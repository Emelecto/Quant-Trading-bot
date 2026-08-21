"""Descarga y cachea datos intradia de BTC para el loop de estrategias.

BTC/USDT perpetual:
  - OHLCV 1h (hasta 3 anos) via ccxt binanceusdm
  - Funding rate historico (cada 8h) via ccxt binanceusdm

Cachea en data/datasets/ para reusar sin red.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(r"C:/venvs/deepfin-repo")
sys.path.insert(0, str(REPO))
import pandas as pd

DATA = REPO / "data" / "datasets"
DATA.mkdir(parents=True, exist_ok=True)


def download_ohlcv_1h(symbol="BTC/USDT", months=36, batch=1000):
    import ccxt
    # SPOT binance (no futures) acepta since sin problema (igual que ensure_raw)
    ex = ccxt.binance({"enableRateLimit": True})
    start = ex.parse8601("2022-01-01T00:00:00Z")
    frames = []
    for _ in range(int((months * 30 * 24) / batch) + 5):
        oh = ex.fetch_ohlcv(symbol, "1h", since=start, limit=batch)
        if not oh:
            break
        df = pd.DataFrame(oh, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        frames.append(df)
        start = oh[-1][0] + 1
        if len(frames) >= 2 and len(frames[-1]) < batch:
            break
    out = pd.concat(frames).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    out.to_csv(DATA / "BTC_USDT_1h.csv", index=False)
    print(f"[ohlcv 1h] filas={len(out)} rango={out['ts'].min()}..{out['ts'].max()}")
    return out


def download_funding(symbol="BTCUSDT", limit=1000, max_calls=60):
    """Funding rate historico via REST directo a fapi.binance.com (evita bug ccxt).
    Itera hacia ADELANTE con startTime para cubrir el rango del 1h (2022-2025).
    """
    import requests
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    frames = []
    start_time = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
    for _ in range(max_calls):
        params = {"symbol": symbol, "limit": limit, "startTime": start_time}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = [data]
            if not data:
                break
            df = pd.DataFrame(data)
            part = pd.DataFrame({
                "ts": pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True),
                "fr": pd.to_numeric(df["fundingRate"]),
            })
            frames.append(part)
            start_time = int(df["fundingTime"].max()) + 1
            if len(frames) >= 2 and len(frames[-1]) < limit:
                break
        except Exception as e:
            print(f"[funding] error en pagina: {e}")
            break
    if not frames:
        print("[funding] sin datos")
        return None
    out = pd.concat(frames).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    out.to_csv(DATA / "BTC_USDT_funding.csv", index=False)
    print(f"[funding] filas={len(out)} rango={out['ts'].min()}..{out['ts'].max()}")
    return out


if __name__ == "__main__":
    print("=== Descargando OHLCV 1h (hasta 3 anos) ===")
    download_ohlcv_1h()
    print("=== Descargando funding rate (opcional) ===")
    try:
        download_funding()
    except Exception as e:
        print(f"[funding] omitido: {e}")
    print("DONE")
