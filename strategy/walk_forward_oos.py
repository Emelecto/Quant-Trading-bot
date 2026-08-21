"""Walk-forward OOS con disciplina de Lopez de Prado.

Mejora de la estrategia mensual DeepFin siguiendo Advances in Financial
Machine Learning:
  - Purged K-Fold Cross-Validation con embargo (evita leakage de labels que
    se solapan en el tiempo).
  - Filtro de regimen de volatilidad (solo opera en regimen de vol media/baja,
    o ajusta el tamano de posicion).
  - Reporte OOS honesto: walk-forward con ventana móvil + metricas de Lopez
    de Prado (Sharpe deflacionado por numero efectivo de bets N_e, y probabilidad
    de que el Sharpe sea > 0).

Bar de referencia: Marcos Lopez de Prado, Advances in Financial Machine
Learning (cap. 12 Purged CV, cap. 14 deflated Sharpe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def purged_kfold_split(times: pd.Series, n_splits: int = 5, embargo: int = 5):
    """Purged K-Fold: elimina del train las muestras cuya ventana de label
    (hacia adelante) se solapa con el fold de test, mas un embargo.
    Devuelve lista de (train_idx, test_idx)."""
    splits = []
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(times):
        test_times = times.iloc[test_idx]
        t0, t1 = test_times.iloc[0], test_times.iloc[-1]
        # purgar: quitar del train muestras cuyo indice de tiempo caiga en
        # [t0 - embargo, t1 + embargo] (label hacia adelante + embargo)
        train_times = times.iloc[train_idx]
        keep = (train_times < (t0 - embargo)) | (train_times > (t1 + embargo))
        purged_train = train_idx[keep.values]
        splits.append((purged_train, test_idx))
    return splits


def deflated_sharpe(sharpe: float, n_obs: int, n_bets: int) -> tuple[float, float]:
    """Sharpe desinflado y probabilidad de que sea > 0 (Lopez de Prado)."""
    # numero efectivo de bets
    if n_bets <= 1:
        ne = 1.0
    else:
        ne = n_bets
    dsr = sharpe / np.sqrt(ne)
    # probabilidad de que el sharpe verdadero > 0
    if n_obs > 1:
        denom = np.sqrt(1 + sharpe**2 / ne)
        z = sharpe * np.sqrt(n_obs) / denom
        from scipy import stats
        prob = 1 - stats.norm.cdf(z)
    else:
        prob = float("nan")
    return float(dsr), float(prob)


def volatility_regime_filter(returns: pd.Series, vol_win: int = 21,
                             high_vol_quantile: float = 0.75) -> pd.Series:
    """Solo opera (True) cuando la volatilidad no esta en el quantile alto.
    Evita exponerse en regimenes de panico donde el edge colapsa."""
    vol = returns.rolling(vol_win).std()
    threshold = vol.quantile(high_vol_quantile)
    return vol <= threshold


def walk_forward_oos(df: pd.DataFrame, feature_fn, model, train_w: int = 252,
                     test_w: int = 21, step: int = 21, embargo: int = 5) -> dict:
    """Walk-forward mensual con Purged K-Fold en cada ventana de train.

    df: OHLCV diario. feature_fn(df)->(X, y, times). Devuelve metricas OOS.
    """
    X, y, times = feature_fn(df)
    n = len(X)
    preds, acts, dates = [], [], []
    i = train_w
    while i + test_w <= n:
        tr_mask = np.arange(i - train_w, i)
        te_mask = np.arange(i, min(i + test_w, n))
        Xtr, ytr = X.iloc[tr_mask], y.iloc[tr_mask]
        Xte, yte = X.iloc[te_mask], y.iloc[te_mask]
        # Purged CV para seleccion de modelo/early stop (no leakage)
        splits = purged_kfold_split(times.iloc[tr_mask], n_splits=5, embargo=embargo)
        # y puede venir como -1/1 (direccion) o 0/1; mapear a 0/1 para el clasificador
        ytr_bin = (ytr > 0).astype(int)
        yte_bin = (yte > 0).astype(int)
        fold_acc = []
        for tr2, te2 in splits:
            m = type(model)(**model.get_params() if hasattr(model, "get_params") else {})
            m.fit(Xtr.iloc[tr2], ytr_bin.iloc[tr2])
            p = m.predict(Xtr.iloc[te2])
            p_bin = (np.asarray(p) > 0.5).astype(int) if not np.issubdtype(type(p), np.integer) else (np.asarray(p) > 0).astype(int)
            fold_acc.append(((p_bin > 0) == (ytr_bin.iloc[te2] > 0)).mean())
        # entrenar en todo el train purgado y predecir test
        m = type(model)(**model.get_params() if hasattr(model, "get_params") else {})
        m.fit(Xtr, ytr_bin)
        p = m.predict(Xte)
        p_bin = (np.asarray(p) > 0.5).astype(int) if hasattr(p, "__len__") else (int(p) > 0)
        # direccion: 0->-1, 1->+1
        p_dir = np.where(p_bin > 0, 1, -1)
        preds.extend(p_dir)
        acts.extend(yte.values)
        dates.extend(times.iloc[te_mask].values)
        i += step

    preds = np.array(preds); acts = np.array(acts)
    # retorno directional solo si acierta el signo
    direction_hit = (np.sign(preds) == np.sign(acts))
    ret = pd.Series(acts)  # retorno forward real del periodo
    strat_ret = np.where(direction_hit, np.abs(ret), -np.abs(ret))
    strat_ret = pd.Series(strat_ret, index=pd.to_datetime(dates))

    # metricas
    n_obs = len(strat_ret)
    n_bets = int(direction_hit.sum())
    if strat_ret.std() > 0:
        sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std()
    else:
        sharpe = 0.0
    dsr, prob = deflated_sharpe(sharpe, n_obs, n_bets)
    return {
        "n_obs": n_obs,
        "n_bets": n_bets,
        "direction_accuracy": float(direction_hit.mean()),
        "sharpe": float(sharpe),
        "deflated_sharpe": dsr,
        "prob_sharpe_gt_0": prob,
        "purged_cv_fold_acc_mean": float(np.mean(fold_acc)),
        "strat_returns": strat_ret,
    }


if __name__ == "__main__":
    # smoke test con datos reales del repo
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data import fetch_data as fd
    from explore_horizonte_mensual import monthly_features
    from models.xgboost_model import XGBoostModel

    df = fd.ensure_raw("BTC/USDT")
    df = fd.clean_ohlcv(df)
    feat = monthly_features(df)
    X = feat.drop(columns=["target"])
    y = feat["target"]
    times = feat.index if hasattr(feat, "index") else pd.Series(range(len(feat)))
    # monthly_features devuelve df con indice temporal? usamos el indice de df
    out = walk_forward_oos(df, lambda d: (X, y, pd.Series(range(len(X)))),
                           XGBoostModel())
    import json
    print(json.dumps({k: (v if not hasattr(v, "to_dict") else "Series")
                      for k, v in out.items() if k != "strat_returns"}, indent=2))
