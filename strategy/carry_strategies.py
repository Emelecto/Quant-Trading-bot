"""Features y estrategias de CARRY para BTC (loop B del gauntlet).

El bar de Lopez de Prado exige edge que sobreviva Purged K-Fold CV con
deflated Sharpe positivo y genere dinero OOS. La direccional mensual ya perdio
(ruido). Las anomalias ESTRUCTURALES de derivados son persistentes:

  1) FUNDING CARRY: estar LONG cuando funding es negativo (cobras funding al
     que esta short), SHORT cuando es positivo. Anomalia documentada.
  2) BASIS CARRY: futuro perpetuo - spot. Contango (futuro > spot) tiende a
     roll-down negativo; backwardation a roll-up positivo.

Estas NO predicen direccion de precio: capturan el pago de carry. Se evaluan
por PnL OOS directo (no modelo ML), lo que el critico ciego compara vs B&H.

Requiere ccxt (Binance futures). Se arreglo el bug de 'since' usando solo limit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_funding_series(symbol: str = "BTC/USDT", limit: int = 1000) -> pd.Series | None:
    """Funding rate historico (diario) via ccxt binanceusdm. Solo limit (evita bug since)."""
    try:
        import ccxt
        ex = ccxt.binanceusdm({"enableRateLimit": True})
        fr = ex.fetch_funding_rate_history(symbol, limit=limit)
        fdf = pd.DataFrame(fr).rename(columns={"datetime": "ts", "fundingRate": "fr"})
        fdf["ts"] = pd.to_datetime(fdf["ts"], utc=True)
        s = fdf.set_index("ts")["fr"].sort_index()
        return s
    except Exception as e:
        print(f"[fetch_funding_series] sin datos: {e}")
        return None


def fetch_basis_series(symbol: str = "BTC/USDT", limit: int = 1000) -> pd.Series | None:
    """Basis = precio futuro perpetuo - spot, via ccxt (markPrice vs last)."""
    try:
        import ccxt
        ex = ccxt.binanceusdm({"enableRateLimit": True})
        # markPrice del perpetual es el precio de futuro
        mp = ex.fetch_mark_ohlcv(symbol, timeframe="1d", limit=limit) if hasattr(ex, "fetch_mark_ohlcv") else None
        if mp is None:
            # fallback: usar close del perpetual como proxy de futuro
            mp = ex.fetch_ohlcv(symbol, "perpetual", timeframe="1d", limit=limit) if False else None
        if mp is None:
            return None
        df = pd.DataFrame(mp, columns=["ts", "o", "h", "l", "c", "v"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")["c"].sort_index()
    except Exception as e:
        print(f"[fetch_basis_series] sin datos: {e}")
        return None


def carry_signal_from_series(series: pd.Series, df: pd.DataFrame,
                             kind: str = "funding") -> pd.Series | None:
    """Senal de carry: +1 (long) si el carry es favorable, -1 (short) si adverso.

    funding: long cuando funding<0 (cobras), short cuando funding>0.
    basis: long cuando basis<0 (backwardation, roll-up), short cuando basis>0.
    """
    if series is None:
        return None
    s = series.reindex(df.index, method="nearest").ffill().dropna()
    if kind == "funding":
        sig = -np.sign(s)  # long si funding negativo
    else:  # basis
        sig = -np.sign(s)
    return pd.Series(sig, index=df.index).reindex(df.index).dropna()


def carry_returns(df: pd.DataFrame, signal: pd.Series, horizon: int = 21) -> pd.Series:
    """Retorno OOS de la estrategia carry SIN solapamiento: senal cada 'horizon'
    dias, mantenida por horizon dias (no compuesto diario irreral).
    """
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    aligned = signal.reindex(df.index)
    # solo operamos en dias multiplos de horizon (posicion no solapada)
    pos_dates = df.index[::horizon]
    ret = aligned * fwd
    ret = ret.where(ret.index.isin(pos_dates), 0.0)
    return ret.dropna()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data import fetch_data as fd

    df = fd.ensure_raw("BTC/USDT"); df = fd.clean_ohlcv(df).set_index("timestamp")

    print("=== FUNDING CARRY ===")
    fr = fetch_funding_series("BTC/USDT")
    if fr is not None:
        sig = carry_signal_from_series(fr, df, "funding")
        ret = carry_returns(df, sig, 21)
        sharpe = np.sqrt(252/21) * ret.mean() / ret.std() if ret.std() > 0 else 0
        eq = (1 + ret.fillna(0)).cumprod()
        print(f"  n={len(ret)} ret_mean={ret.mean():.5f} sharpe_anual={sharpe:.3f} "
              f"eq_final={eq.iloc[-1]:.3f} vs B&H={(1+df['close'].pct_change(21).shift(-21).dropna()).cumprod().iloc[-1]:.3f}")
    else:
        print("  sin datos de funding")

    print("=== BASIS CARRY ===")
    basis = fetch_basis_series("BTC/USDT")
    if basis is not None:
        sig = carry_signal_from_series(basis, df, "basis")
        ret = carry_returns(df, sig, 21)
        sharpe = np.sqrt(252/21) * ret.mean() / ret.std() if ret.std() > 0 else 0
        eq = (1 + ret.fillna(0)).cumprod()
        print(f"  n={len(ret)} ret_mean={ret.mean():.5f} sharpe_anual={sharpe:.3f} eq_final={eq.iloc[-1]:.3f}")
    else:
        print("  sin datos de basis")
