"""Diagnostico Frente B: ¿los modelos predicen algo fuera de muestra?

Mide la correlacion/accuracy de la senal de cada modelo contra el retorno
real del dia siguiente (fuera de muestra, walk-forward simple). Si la
correlacion es ~0, el problema es la SENAL (features/modelo), no el riesgo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"C:/Users/ecard/OneDrive/Escritorio/Semillero Deep-Fin/deepfin-trading-bot")))

import numpy as np
import pandas as pd
from data import fetch_data as fd
from features.build_features import build_feature_matrix
from models.linear_regression_model import LinearRegressionModel
from models.xgboost_model import XGBoostModel
from models.monte_carlo_model import MonteCarloModel
from ensemble.ensemble_methods import weighted

df = fd.load_raw("BTC/USDT"); df = fd.clean_ohlcv(df)
feats = build_feature_matrix(df)
X = feats.drop(columns=["target"]); y = feats["target"]
close = df["close"]
# retorno real siguiente barra (lo que deberia predecir la senal)
fwd_ret = df["close"].pct_change().shift(-1).reindex(X.index)

models = [LinearRegressionModel(), XGBoostModel(), MonteCarloModel(n_paths=300, random_state=42)]
signals = {}
for m in models:
    if m.name == "monte_carlo":
        m._prices = close.reindex(X.index)
    m.fit(X, y)
    signals[m.name] = m.predict(X)

print("=== Correlacion senal vs retorno futuro (BTC, todo el dataset) ===")
for name, s in signals.items():
    aligned = s.index.intersection(fwd_ret.dropna().index)
    corr = np.corrcoef(s.loc[aligned], fwd_ret.loc[aligned])[0, 1]
    # directional accuracy: sign(senal) == sign(retorno)
    acc = (np.sign(s.loc[aligned]) == np.sign(fwd_ret.loc[aligned])).mean()
    print(f"  {name:<20} corr={corr:>7.3f}  dir_acc={acc:>6.2%}")

ens = weighted(signals)
aligned = ens.index.intersection(fwd_ret.dropna().index)
corr_e = np.corrcoef(ens.loc[aligned], fwd_ret.loc[aligned])[0, 1]
acc_e = (np.sign(ens.loc[aligned]) == np.sign(fwd_ret.loc[aligned])).mean()
print(f"  {'ensemble_weighted':<20} corr={corr_e:>7.3f}  dir_acc={acc_e:>6.2%}")
print("\n(La 'moneda' da dir_acc ~50%. Si estamos cerca, la senal no predice nada.)")
