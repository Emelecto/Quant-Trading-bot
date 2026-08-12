"""
Dashboard Streamlit para el bot DeepFin (visualización, paper trading).

Muestra: señal actual del ensemble, equity curve simulada y métricas
comparativas (modelos/ensemble vs Buy & Hold). No controla el bot en vivo (MVP).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_data as fd
from features.build_features import build_feature_matrix
from models.linear_regression_model import LinearRegressionModel
from models.xgboost_model import XGBoostModel
from models.monte_carlo_model import MonteCarloModel
from ensemble.ensemble_methods import weighted
from backtest import metrics as M


st.set_page_config(page_title="DeepFin Bot", layout="wide")
st.title("DeepFin — Bot de Trading Ensemble (Paper Trading)")

SYMBOL = st.sidebar.selectbox("Activo", ["BTC/USDT", "ETH/USDT"], index=0)

try:
    df = fd.load_raw(SYMBOL)
except FileNotFoundError:
    st.error(f"Sin datos para {SYMBOL}. Ejecuta: python -m data.fetch_data")
    st.stop()

df = fd.clean_ohlcv(df)
feats = build_feature_matrix(df)
X = feats.drop(columns=["target"])
y = feats["target"]
close = df["close"]

models = [LinearRegressionModel(), XGBoostModel(), MonteCarloModel(n_paths=300)]
signals = {}
for m in models:
    if m.name == "monte_carlo":
        m._prices = close.reindex(X.index)
    m.fit(X, y)
    signals[m.name] = m.predict(X)
final_signal = weighted(signals)

st.metric("Señal del ensemble (score)", f"{final_signal.iloc[-1]:.3f}",
          help=">0 compra, <0 venta, ~0 neutral")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Equity Curve (señal acumulada)")
    eq = (1 + final_signal.shift(1).fillna(0) * df["close"].pct_change().reindex(final_signal.index).fillna(0)).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=eq.values, mode="lines", name="Ensemble"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Componentes de la señal")
    comp = pd.DataFrame(signals)
    st.line_chart(comp)

st.subheader("Métricas comparativas")
bh_rets = df["close"].pct_change().dropna()
bh_eq = (1 + bh_rets).cumprod()
ens_rets = final_signal.shift(1).fillna(0) * df["close"].pct_change().reindex(final_signal.index).fillna(0)
ens_eq = (1 + ens_rets.fillna(0)).cumprod()
m_ens = M.all_metrics(ens_rets.fillna(0), ens_eq)
m_bh = M.all_metrics(bh_rets, bh_eq)
cmp = pd.DataFrame({"Ensemble": m_ens, "Buy & Hold": m_bh}).T
st.dataframe(cmp.style.format("{:.3f}"))
st.caption("Proyecto educativo/investigativo. No constituye asesoría financiera ni garantía de rentabilidad.")
