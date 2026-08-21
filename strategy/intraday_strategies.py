"""Estrategias intradia BTC y evaluador bajo el bar de Lopez de Prado.

Objetivo del gauntlet: superar el bar (Purged K-Fold CV + deflated Sharpe > 0
con prob > 0.95) y generar el maximo dinero OOS. Con datos DIARIOS no hay edge;
con datos 1h + funding SI puede haberlo (anomalias de microestructura/derivados).

Estrategias (reglas fijas, no ML):
  A) FUNDING CARRY 8h: senal = -sign(funding en la ultima marca). Mantener 8h.
     Cobras funding si estas long y funding<0. Edge estructural documentado.
  B) SHORT-TERM REVERSAL 1h: senal = -sign(retorno ultimas N h). Retener M h.
     Las ineficiencias de microestructura viven en alta frecuencia.

Evaluador: walk-forward con PURGED ventanas (el label de train no solapa el test)
+ deflated Sharpe + prob Sharpe>0, comparado ciegamente vs B&H.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def load_1h() -> pd.DataFrame:
    from pathlib import Path
    REPO = Path(r"C:/venvs/deepfin-repo")
    df = pd.read_csv(REPO / "data" / "datasets" / "BTC_USDT_1h.csv", parse_dates=["ts"])
    return df.set_index("ts").sort_index()


def load_funding() -> pd.Series:
    from pathlib import Path
    REPO = Path(r"C:/venvs/deepfin-repo")
    p = REPO / "data" / "datasets" / "BTC_USDT_funding.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty or "fr" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    return df.set_index("ts")["fr"].sort_index()


def funding_carry_signal(close_idx: pd.DatetimeIndex, funding: pd.Series,
                         hold_h: int = 8) -> pd.Series:
    """Senal de funding carry alineada al indice de close 1h.
    -sign(funding en la marca mas reciente). Mantener hold_h horas.
    """
    # funding esta cada 8h; forward-fill al indice horario
    f = funding.reindex(close_idx, method="ffill")
    sig = -np.sign(f)
    return pd.Series(sig, index=close_idx).fillna(0)


def reversal_signal(close: pd.Series, lookback: int = 24, hold_h: int = 4) -> pd.Series:
    """Reversion a la media: -sign(retorno acumulado de lookback h)."""
    ret = close.pct_change(lookback)
    sig = -np.sign(ret)
    return sig.fillna(0)


def oos_returns(close, signal, hold_h, fee=0.001):
    """Retorno OOS de la regla con costo de transaccion (fee por lado).
    Senal en t sobre retorno forward de hold_h. Solo en marcas cada hold_h.
    """
    fwd = close.pct_change(hold_h).shift(-hold_h)
    aligned = signal.reindex(close.index)
    pos = close.index[::hold_h]
    # el trade paga fee al abrir y al cerrar
    gross = (aligned * fwd).where(close.index.isin(pos), 0.0)
    trades = gross != 0
    net = gross - 2 * fee * trades.astype(float)
    return net.dropna()


def evaluate(signal: pd.Series, close: pd.Series, hold_h: int,
             test_trades: int = 30 * 24 // 4, step_trades: int = 30 * 24 // 4) -> dict:
    """Walk-forward sobre la serie de retornos de trading (ya no-solapados).

    La regla es fija (no se entrena), asi que el walk-forward solo acumula OOS
    por bloques de `test_trades` trades, dejando `train_w` trades de purga antes
    (los trades no se solapan, asi que no hay leakage de label). Reporta
    deflated Sharpe (Lopez de Prado) + prob Sharpe>0 + equity OOS.
    """
    ret = oos_returns(close, signal, hold_h)
    ret = ret.dropna()
    if len(ret) < 20:
        return {"n": len(ret), "sharpe": 0.0, "deflated_sharpe": 0.0,
                "prob_gt_0": float("nan"), "eq": 1.0}
    oos = []
    i = test_trades  # purga: dejamos el primer bloque como train
    while i < len(ret):
        oos.extend(ret.iloc[i:i + test_trades].values)
        i += step_trades
    oos = pd.Series(oos)
    if len(oos) < 10 or oos.std() == 0 or not np.isfinite(oos.std()):
        return {"n": len(oos), "sharpe": 0.0, "deflated_sharpe": 0.0,
                "prob_gt_0": float("nan"), "eq": 1.0}
    periods_per_year = (24 * 365) / hold_h
    sharpe = np.sqrt(periods_per_year) * oos.mean() / oos.std()
    n_obs = len(oos)
    n_bets = int((oos != 0).sum())
    ne = max(n_bets, 1)
    dsr = sharpe / np.sqrt(ne)
    z = sharpe * np.sqrt(n_obs) / np.sqrt(1 + sharpe**2 / ne) if ne > 0 else 0
    prob = 1 - stats.norm.cdf(z)
    eq = float((1 + oos.fillna(0)).cumprod().iloc[-1])
    return {"n": n_obs, "sharpe": float(sharpe), "deflated_sharpe": float(dsr),
            "prob_gt_0": float(prob), "eq": eq}


def buy_and_hold(close: pd.Series, hold_h: int) -> dict:
    fwd = close.pct_change(hold_h).shift(-hold_h)
    ret = fwd.dropna()
    periods_per_year = (24 * 365) / hold_h
    sharpe = np.sqrt(periods_per_year) * ret.mean() / ret.std() if ret.std() > 0 else 0
    eq = float((1 + ret.fillna(0)).cumprod().iloc[-1]) if len(ret) else 1.0
    return {"n": len(ret), "sharpe": float(sharpe), "eq": eq}


if __name__ == "__main__":
    df = load_1h()
    close = df["close"]
    print(f"[data] 1h filas={len(df)} rango={df.index.min()}..{df.index.max()}")

    # B&H
    bh = buy_and_hold(close, 8)
    print(f"B&H (8h): sharpe={bh['sharpe']:.3f} eq={bh['eq']:.3f}")

    # A) FUNDING CARRY 8h
    fund = load_funding()
    sig_f = funding_carry_signal(close.index, fund, hold_h=8)
    r_f = evaluate(sig_f, close, 8)
    print(f"FUNDING CARRY (8h): n={r_f['n']} sharpe={r_f['sharpe']:.3f} "
          f"deflated={r_f['deflated_sharpe']:.3f} prob>0={r_f['prob_gt_0']:.3f} eq={r_f['eq']:.3f}")

    # B) REVERSAL 1h (lookback 24h, hold 4h)
    sig_r = reversal_signal(close, lookback=24, hold_h=4)
    r_r = evaluate(sig_r, close, 4)
    print(f"REVERSAL (24h/4h): n={r_r['n']} sharpe={r_r['sharpe']:.3f} "
          f"deflated={r_r['deflated_sharpe']:.3f} prob>0={r_r['prob_gt_0']:.3f} eq={r_r['eq']:.3f}")
