"""
Dashboard Streamlit para el bot DeepFin (visualización, paper trading).

Muestra: señal actual del ensemble, equity curve simulada (con capa de riesgo
SL/TP + filtro de régimen vía motor walk-forward) y métricas comparativas
(ensemble vs Buy & Hold). No controla el bot en vivo (MVP).

El 3er gráfico y la tabla de métricas usan el MISMO motor de backtest con
riesgo (SL=2*ATR, TP 1:3, filtro ADX) para ser coherentes con el informe.
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
from risk import risk_management as rm
from backtest.engine import run_walk_forward
from backtest import metrics as M


# Parámetros de riesgo (coherentes con el backtest y el informe)
ATR_MULT = 2.0
RR_RATIO = 3.0
ADX_THRESHOLD = 20.0
TRAIN_WINDOW = 252
TEST_WINDOW = 63
STEP = 63


@st.cache_data(show_spinner="Entrenando modelos y corriendo backtest con riesgo...")
def compute_backtest(SYMBOL: str) -> dict:
    """Corre el walk-forward con riesgo UNA vez por activo (cache de Streamlit)."""
    df = fd.load_raw(SYMBOL)
    df = fd.clean_ohlcv(df)

    models = [
        LinearRegressionModel(),
        XGBoostModel(),
        MonteCarloModel(n_paths=300, random_state=42),
    ]
    # retornos del ensemble (walk-forward, con SL/TP + régimen)
    ens_rets = run_walk_forward(
        df, models, ensemble_method="weighted",
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW, step=STEP,
        atr_mult=ATR_MULT, rr_ratio=RR_RATIO, adx_threshold=ADX_THRESHOLD,
    )
    # benchmark buy & hold (mismo periodo que el ensemble)
    bh = df["close"].pct_change().reindex(ens_rets.index).dropna()
    ens_rets = ens_rets.reindex(bh.index)  # alinear al periodo común

    ens_eq = (1 + ens_rets.fillna(0)).cumprod()
    bh_eq = (1 + bh.fillna(0)).cumprod()

    # señal actual del ensemble (última ventana, para el indicador)
    feats = build_feature_matrix(df)
    X = feats.drop(columns=["target"]); y = feats["target"]; close = df["close"]
    last_signal = weighted({m.name: _train_predict(m, X, y, close) for m in models}).iloc[-1]

    return {
        "ens_rets": ens_rets, "bh": bh,
        "ens_eq": ens_eq, "bh_eq": bh_eq,
        "last_signal": float(last_signal),
        "m_ens": M.all_metrics(ens_rets.fillna(0), ens_eq),
        "m_bh": M.all_metrics(bh, bh_eq),
    }


def _train_predict(m, X, y, close):
    """Entrena y predice en TODO el dataset (solo para la señal actual del dashboard)."""
    if getattr(m, "estimator", None) is not None or hasattr(m, "fit"):
        inst = type(m)(**{k: v for k, v in vars(m).items()
                          if k in __import__("inspect").signature(type(m).__init__).parameters and k != "self"})
        if m.name == "monte_carlo":
            inst._prices = close.reindex(X.index)
        inst.fit(X, y)
        return inst.predict(X)
    return m.predict(X)


st.set_page_config(page_title="DeepFin Bot", layout="wide")
st.title("DeepFin — Bot de Trading Ensemble (Paper Trading)")

SYMBOL = st.sidebar.selectbox("Activo", ["BTC/USDT", "ETH/USDT"], index=0)

try:
    res = compute_backtest(SYMBOL)
except FileNotFoundError:
    st.error(f"Sin datos para {SYMBOL}. Ejecuta: python -m data.fetch_data")
    st.stop()

st.metric("Señal del ensemble (score)", f"{res['last_signal']:.3f}",
          help=">0 compra, <0 venta, ~0 neutral (señal de la última ventana)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Equity Curve (señal acumulada, sin riesgo)")
    feats = build_feature_matrix(fd.load_raw(SYMBOL))
    X = feats.drop(columns=["target"]); y = feats["target"]
    close = fd.clean_ohlcv(fd.load_raw(SYMBOL))["close"]
    models = [LinearRegressionModel(), XGBoostModel(), MonteCarloModel(n_paths=300)]
    signals = {}
    for m in models:
        if m.name == "monte_carlo":
            m._prices = close.reindex(X.index)
        m.fit(X, y)
        signals[m.name] = m.predict(X)
    final_signal = weighted(signals)
    eq_simple = (1 + final_signal.shift(1).fillna(0) * fd.clean_ohlcv(fd.load_raw(SYMBOL))["close"]
                 .pct_change().reindex(final_signal.index).fillna(0)).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=eq_simple.values, mode="lines", name="Ensemble (simple)"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Componentes de la señal")
    comp = pd.DataFrame(signals)
    st.line_chart(comp)

# === 3er gráfico: Equity Curve Ensemble (con riesgo) vs Buy & Hold ===
st.subheader("Equity Curve: Ensemble (con riesgo) vs Buy & Hold")
ens_eq = res["ens_eq"] / res["ens_eq"].iloc[0]
bh_eq = res["bh_eq"] / res["bh_eq"].iloc[0]
cmp_eq = pd.DataFrame({"Ensemble (riesgo)": ens_eq, "Buy & Hold": bh_eq}).dropna()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=cmp_eq["Ensemble (riesgo)"].values, mode="lines", name="Ensemble (riesgo)"))
fig2.add_trace(go.Scatter(y=cmp_eq["Buy & Hold"].values, mode="lines", name="Buy & Hold"))
fig2.update_layout(yaxis_title="Capital normalizado (1.0 = inicio)", xaxis_title="Fecha",
                   legend=dict(orientation="h"), hovermode="x unified")
st.plotly_chart(fig2, use_container_width=True)
st.caption("Curva del ensemble calculada con el motor walk-forward y capa de riesgo "
           "(SL=2*ATR, TP 1:3, filtro ADX) — coherente con el backtest del informe. "
           "Si la del ensemble queda por debajo, el bot NO superó a no hacer nada en ese periodo.")

st.subheader("Métricas comparativas")
cmp = pd.DataFrame({"Ensemble (riesgo)": res["m_ens"], "Buy & Hold": res["m_bh"]}).T
st.dataframe(cmp.style.format("{:.3f}"))
st.caption("Proyecto educativo/investigativo. No constituye asesoría financiera ni garantía de rentabilidad.")

# === Paper Trading en Vivo (BTC mensual, señal operativa) ===
st.divider()
st.subheader("📡 Paper Trading en Vivo — BTC mensual (riesgo 1%)")
st.caption("Senal operativa del modelo mensual con los datos mas recientes de Binance. "
           "Capital simulado (paper). No ejecuta ordenes reales.")
if st.button("Generar senal mensual actual"):
    try:
        with st.spinner("Entrenando modelo mensual y generando senal..."):
            from live import paper_trader as pt
            sig = pt.run(capital=50.0, risk=0.01)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Direccion", sig["direction"], help="LONG / SHORT / NEUTRAL")
        col_b.metric("Score senal", f"{sig['signal_score']:.3f}")
        col_c.metric("Tamano posicion", f"{sig['position_size_pct']:.1f}%")
        st.write(f"**Cierre actual:** ${sig['last_close']:,.2f}  |  **SL:** ${sig['sl_level']:,.2f}  |  **TP:** ${sig['tp_level']:,.2f}")
        st.write(f"**Volatilidad 21d:** {sig['vol_21d']:.4f}  |  **Riesgo por trade:** {sig['risk_per_trade_pct']:.1f}%  |  **Mantener:** {sig['hold_days']} dias")
        st.json(sig)
    except Exception as e:
        st.error(f"No se pudo generar la senal en vivo (¿sin acceso a Binance?): {e}")
else:
    st.info("Haz clic en 'Generar senal mensual actual' para ver la operacion sugerida (paper).")
