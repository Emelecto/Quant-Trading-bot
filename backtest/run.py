"""
Script principal de backtest (walk-forward) para BTC y ETH.

Carga datos limpios, corre el ensemble por activo y reporta métricas
comparativas (modelos individuales vs ensemble) y vs benchmark Buy & Hold.

Uso: python -m backtest.run
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from features.build_features import build_feature_matrix
from models.linear_regression_model import LinearRegressionModel
from models.xgboost_model import XGBoostModel
from models.monte_carlo_model import MonteCarloModel
from backtest.engine import run_walk_forward
from backtest import metrics as M


def buy_and_hold_benchmark(df: pd.DataFrame) -> pd.Series:
    """Retorno de mantener el activo durante todo el periodo (diario)."""
    return df["close"].pct_change().dropna()


def main():
    models = [
        LinearRegressionModel(),
        XGBoostModel(),
        MonteCarloModel(n_paths=500, random_state=42),
    ]
    for method in ["voting", "weighted"]:
        print(f"\n===== ENSEMBLE: {method.upper()} =====")
        for sym in ["BTC/USDT", "ETH/USDT"]:
            try:
                df = fd.load_raw(sym)
            except FileNotFoundError:
                print(f"[{sym}] sin datos locales; descargando...")
                df = fd.fetch_and_persist(symbol=sym, lookback_years=3)
            df = fd.clean_ohlcv(df)

            # métricas de modelos individuales (señal -> retorno simple siguiente barra)
            feats = build_feature_matrix(df)

            # ensemble walk-forward
            rets = run_walk_forward(
                df, models, ensemble_method=method,
                train_window=252, test_window=63, step=63,
            )
            eq = (1 + rets.fillna(0)).cumprod()
            ind = M.all_metrics(rets, eq)

            # benchmark buy & hold
            bh = buy_and_hold_benchmark(df).reindex(rets.index).dropna()
            bh_eq = (1 + bh.fillna(0)).cumprod()
            bh_metrics = M.all_metrics(bh, bh_eq)

            print(f"\n[{sym}] ENSEMBLE ({method})")
            print(f"  Sharpe: {ind['sharpe']:.3f} | Sortino: {ind['sortino']:.3f} | "
                  f"MaxDD: {ind['max_drawdown']:.2%} | Win: {ind['win_rate']:.2%} | "
                  f"PF: {ind['profit_factor']:.2f} | CAGR: {ind['cagr']:.2%}")
            print(f"[{sym}] BUY&HOLD")
            print(f"  Sharpe: {bh_metrics['sharpe']:.3f} | Sortino: {bh_metrics['sortino']:.3f} | "
                  f"MaxDD: {bh_metrics['max_drawdown']:.2%} | Win: {bh_metrics['win_rate']:.2%} | "
                  f"PF: {bh_metrics['profit_factor']:.2f} | CAGR: {bh_metrics['cagr']:.2%}")


if __name__ == "__main__":
    main()
