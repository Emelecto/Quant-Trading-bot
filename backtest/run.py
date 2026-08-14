"""
Script principal de backtest (walk-forward) para BTC y ETH.

Reporta la comparación head-to-head que exige la investigación:
  modelo individual (LR / XGBoost / Monte Carlo / CatBoost)
  vs ensemble (voting / weighted / stacking)
  vs benchmark Buy & Hold.

Uso:
  python -m backtest.run
  python -m backtest.run --garch      # usa SL basado en volatilidad GARCH
  python -m backtest.run --catboost    # incluye modelo CatBoost en el ensemble
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from features.build_features import build_feature_matrix
from models.linear_regression_model import LinearRegressionModel
from models.xgboost_model import XGBoostModel
from models.monte_carlo_model import MonteCarloModel
from models.catboost_model import CatBoostModel
from backtest.engine import run_walk_forward
from backtest import metrics as M


def buy_and_hold_benchmark(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().dropna()


def print_comparison(title: str, rows: dict[str, pd.Series]):
    """Imprime tabla comparativa de métricas para varias series de retornos."""
    print(f"\n=== {title} ===")
    header = f"{'Estrategia':<14}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}{'Win%':>8}{'PF':>7}{'CAGR':>9}"
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        eq = (1 + r.fillna(0)).cumprod()
        m = M.all_metrics(r, eq)
        print(f"{name:<14}{m['sharpe']:>9.3f}{m['sortino']:>9.3f}{m['max_drawdown']:>9.2%}"
              f"{m['win_rate']:>8.2%}{m['profit_factor']:>7.2f}{m['cagr']:>9.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garch", action="store_true", help="usar SL por volatilidad GARCH")
    ap.add_argument("--catboost", action="store_true", help="incluir CatBoost en el ensemble")
    args = ap.parse_args()

    models = [
        LinearRegressionModel(),
        XGBoostModel(),
        MonteCarloModel(n_paths=300, random_state=42),
    ]
    if args.catboost:
        models.append(CatBoostModel(random_state=42))

    for sym in ["BTC/USDT", "ETH/USDT"]:
        try:
            df = fd.load_raw(sym)
        except FileNotFoundError:
            print(f"[{sym}] sin datos locales; descargando...")
            df = fd.fetch_and_persist(symbol=sym, lookback_years=3)
        df = fd.clean_ohlcv(df)

        rows: dict[str, pd.Series] = {}

        # --- modelos individuales (head-to-head) ---
        indiv = run_walk_forward(df, models, ensemble_method="weighted",
                                 return_individual=True, use_garch_sl=args.garch)
        for name, s in indiv["models"].items():
            rows[name] = s

        # --- ensemble: voting / weighted / stacking ---
        for method in ["voting", "weighted", "stacking"]:
            try:
                r = run_walk_forward(df, models, ensemble_method=method,
                                     return_individual=False, use_garch_sl=args.garch)
                rows[f"ensemble_{method}"] = r
            except Exception as e:
                print(f"  [{sym}] ensemble {method} error: {e}")

        # --- benchmark ---
        bh = buy_and_hold_benchmark(df)
        rows["buy_and_hold"] = bh

        print_comparison(f"{sym}  (GARCH_SL={args.garch}, CatBoost={args.catboost})", rows)


if __name__ == "__main__":
    main()
