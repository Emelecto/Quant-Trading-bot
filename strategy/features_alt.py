"""Features alternativas para la estrategia BTC (loop B del gauntlet).

El bar de Lopez de Prado exige un edge que sobreviva Purged K-Fold CV con
deflated Sharpe positivo. La senal mensual de momentum/vol-state NO sobrevive
(deflated Sharpe 0.13, accuracy 26%). Buscamos anomalias ESTRUCTURALES de BTC
que la literatura y la practica muestran persistentes:

  1) Funding rate (ccxt): funding muy negativo -> retorno positivo subsecuente
     (shorts sobreapalancados se liquidan); funding muy positivo -> lo contrario.
  2) Open Interest + precio: OI sube con precio (crowding long) suele anteceder
     correccion; OI cae en caida de precio -> capitulacion (rebote).
  3) Basis (futuro - spot): contango extremo -> roll-down negativo.
  4) Reversion a la media intradia (microestructura): retornos cortos revierten.

Cada feature devuelve (X, y, times) listo para walk_forward_oos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _monthly_target(df: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Retorno forward de horizon dias (target direccional)."""
    ret = df["close"].pct_change(horizon).shift(-horizon)
    return np.sign(ret).fillna(0)


def microstructure_features(df: pd.DataFrame, horizon: int = 21) -> tuple:
    """Features de microestructura y tecnica sobre OHLCV diario."""
    close = df["close"]
    ret = close.pct_change()
    # reversión a la media de retornos cortos
    rev_5 = -ret.rolling(5).sum()          # si subió 5d, espera caída
    rev_21 = -ret.rolling(horizon).sum()
    # breakout de volatilidad: rango reciente vs histórico
    rng = (df["high"] - df["low"]) / close
    breakout = rng.rolling(5).mean() / rng.rolling(horizon).mean() - 1
    # momentum de volumen (crowding)
    vol_chg = df["volume"].pct_change().rolling(5).mean()
    # sesgo: retorno acumulado reciente normalizado
    roll_z = (close / close.rolling(horizon).mean() - 1)
    X = pd.DataFrame({
        "rev_5": rev_5, "rev_21": rev_21, "breakout": breakout,
        "vol_chg": vol_chg, "roll_z": roll_z,
    }).dropna()
    y = _monthly_target(df, horizon).reindex(X.index)
    X = X.join(y.rename("target")).dropna()
    y = X.pop("target")
    return X, y, pd.Series(range(len(X)))


def funding_features(symbol: str = "BTC/USDT", horizon: int = 21) -> tuple | None:
    """Features de funding rate + open interest via ccxt (Binance).

    Requiere red. Devuelve None si no hay datos.
    """
    try:
        import ccxt
        ex = ccxt.binanceusdm({"enableRateLimit": True})  # futuros USDT-M (tiene funding/OI)
        since = int((pd.Timestamp.utcnow() - pd.DateOffset(years=3)).timestamp() * 1000)
        fr = ex.fetch_funding_rate_history("BTC/USDT", since=since, limit=1000)
        oi = ex.fetch_open_interest_history("BTC/USDT", timeframe="1d", since=since, limit=1000)
    except Exception as e:
        print(f"[funding_features] sin datos ccxt: {e}")
        return None
    fdf = pd.DataFrame(fr).rename(columns={"datetime": "ts", "fundingRate": "fr"})
    fdf["ts"] = pd.to_datetime(fdf["ts"], utc=True)
    odf = pd.DataFrame(oi).rename(columns={"datetime": "ts", "openInterest": "oi"})
    odf["ts"] = pd.to_datetime(odf["ts"], utc=True)
    m = fdf.merge(odf, on="ts", how="outer").set_index("ts").sort_index()
    m["fr"] = m["fr"].ffill()
    m["oi"] = m["oi"].ffill()
    # features: funding extremo y cambio de OI
    m["fr_z"] = (m["fr"] - m["fr"].rolling(horizon).mean()) / (m["fr"].rolling(horizon).std() + 1e-9)
    m["oi_chg"] = m["oi"].pct_change(horizon)
    # necesitamos el target direccional usando close de la misma fecha
    # (el caller debe pasar df con close para alinear)
    return m[["fr", "fr_z", "oi_chg"]]


def build_funding_with_price(df: pd.DataFrame, symbol: str = "BTC/USDT",
                             horizon: int = 21) -> tuple | None:
    """Une funding/oi con el close de df para armar (X, y, times)."""
    m = funding_features(symbol, horizon)
    if m is None:
        return None
    m = m.copy()
    m["close"] = df["close"].reindex(m.index, method="nearest")
    m = m.dropna()
    if len(m) < horizon * 2:
        return None
    y = _monthly_target(df, horizon).reindex(m.index)
    m = m.join(y.rename("target")).dropna()
    y = m.pop("target")
    return m.drop(columns=["close"]), y, pd.Series(range(len(m)))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data import fetch_data as fd
    from strategy.walk_forward_oos import walk_forward_oos
    from models.xgboost_model import XGBoostModel

    df = fd.ensure_raw("BTC/USDT"); df = fd.clean_ohlcv(df)

    print("=== Microestructura (sin red) ===")
    X, y, times = microstructure_features(df)
    out = walk_forward_oos(df, lambda d: (X, y, times), XGBoostModel())
    print(f"  n_obs={out['n_obs']} acc={out['direction_accuracy']:.3f} "
          f"deflated_sharpe={out['deflated_sharpe']:.3f} prob>0={out['prob_sharpe_gt_0']:.3f}")

    print("=== Funding + OI (ccxt red) ===")
    res = build_funding_with_price(df)
    if res:
        X2, y2, t2 = res
        out2 = walk_forward_oos(df, lambda d: (X2, y2, t2), XGBoostModel())
        print(f"  n_obs={out2['n_obs']} acc={out2['direction_accuracy']:.3f} "
              f"deflated_sharpe={out2['deflated_sharpe']:.3f} prob>0={out2['prob_sharpe_gt_0']:.3f}")
    else:
        print("  sin datos de funding/oi disponibles")
