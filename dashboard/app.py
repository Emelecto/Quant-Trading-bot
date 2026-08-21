"""Dashboard Streamlit para el bot DeepFin (visualización, paper trading).

Muestra: señal actual del ensemble, equity curve simulada (con capa de riesgo
SL/TP + filtro de régimen vía motor walk-forward) y métricas comparativas
(ensemble vs Buy & Hold). Incluye paper trading en vivo mensual (BTC, riesgo 1%)
y el historial de trades del paper broker local (ledger_bot).

Cloud-safe: los datos vienen de data/datasets/ (tracked en el repo), asi
funciona en Streamlit Cloud sin ccxt ni descarga de red.
"""
from __future__ import annotations

import sys
import json
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
from backtest.engine import run_walk_forward
from backtest import metrics as M


# Parámetros de riesgo (coherentes con el backtest y el informe)
ATR_MULT = 2.0
RR_RATIO = 3.0
ADX_THRESHOLD = 20.0
TRAIN_WINDOW = 252
TEST_WINDOW = 63
STEP = 63

LEDGER_PATH = Path(__file__).resolve().parent.parent / "live" / "trades.json"


@st.cache_data(show_spinner="Entrenando modelos y corriendo backtest con riesgo...")
def compute_backtest(SYMBOL: str) -> dict:
    """Corre el walk-forward con riesgo UNA vez por activo (cache de Streamlit)."""
    try:
        if not SYMBOL:
            SYMBOL = "BTC/USDT"
        df = fd.ensure_raw(SYMBOL)
        df = fd.clean_ohlcv(df)

        models = [
            LinearRegressionModel(),
            XGBoostModel(),
            MonteCarloModel(n_paths=300, random_state=42),
        ]
        ens_rets = run_walk_forward(
            df, models, ensemble_method="weighted",
            train_window=TRAIN_WINDOW, test_window=TEST_WINDOW, step=STEP,
            atr_mult=ATR_MULT, rr_ratio=RR_RATIO, adx_threshold=ADX_THRESHOLD,
        )
        bh = df["close"].pct_change().reindex(ens_rets.index).dropna()
        ens_rets = ens_rets.reindex(bh.index)

        ens_eq = (1 + ens_rets.fillna(0)).cumprod()
        bh_eq = (1 + bh.fillna(0)).cumprod()

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
    except Exception as e:
        return {"error": f"No se pudo correr el backtest para {SYMBOL}: {e}"}


def _train_predict(m, X, y, close):
    """Entrena y predice en TODO el dataset (solo para la señal actual del dashboard)."""
    import inspect
    if getattr(m, "estimator", None) is not None or hasattr(m, "fit"):
        inst = type(m)(**{k: v for k, v in vars(m).items()
                          if k in inspect.signature(type(m).__init__).parameters and k != "self"})
        if m.name == "monte_carlo":
            inst._prices = close.reindex(X.index)
        inst.fit(X, y)
        return inst.predict(X)
    return m.predict(X)


def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return {"capital": 50.0, "trades": [], "open_position": None}


st.set_page_config(page_title="DeepFin Bot", layout="wide")
st.title("DeepFin — Bot de Trading Ensemble (Paper Trading)")

SYMBOL = st.sidebar.selectbox("Activo", ["BTC/USDT", "ETH/USDT"], index=0)

try:
    res = compute_backtest(SYMBOL)
except Exception as e:
    st.error(f"Error al cargar el backtest: {e}")
    st.stop()

if res.get("error"):
    st.error(res["error"])
    st.stop()

# Pestañas: Backtest | Paper Broker (trades)
tab1, tab2 = st.tabs(["📊 Backtest Ensemble", "📜 Historial de Trades (Paper Broker)"])

with tab1:
    st.metric("Señal del ensemble (score)", f"{res['last_signal']:.3f}",
              help=">0 compra, <0 venta, ~0 neutral (señal de la última ventana)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Equity Curve (señal acumulada, sin riesgo)")
        feats = build_feature_matrix(fd.ensure_raw(SYMBOL))
        X = feats.drop(columns=["target"]); y = feats["target"]
        close = fd.clean_ohlcv(fd.ensure_raw(SYMBOL))["close"]
        models = [LinearRegressionModel(), XGBoostModel(), MonteCarloModel(n_paths=300)]
        signals = {}
        for m in models:
            if m.name == "monte_carlo":
                m._prices = close.reindex(X.index)
            m.fit(X, y)
            signals[m.name] = m.predict(X)
        final_signal = weighted(signals)
        eq_simple = (1 + final_signal.shift(1).fillna(0) * fd.clean_ohlcv(fd.ensure_raw(SYMBOL))["close"]
                     .pct_change().reindex(final_signal.index).fillna(0)).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=eq_simple.values, mode="lines", name="Ensemble (simple)"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Componentes de la señal")
        comp = pd.DataFrame(signals)
        st.line_chart(comp)

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

with tab2:
    st.subheader("📜 Historial de Trades — Paper Broker Local")
    st.caption("Posiciones mensuales BTC con marca a mercado diaria (precio real de Binance) "
               "y ledger persistente. $0 real. Modelo freqtrade Trades tab.")
    ledger = load_ledger()
    cap = ledger.get("capital", 50.0)
    trades = ledger.get("trades", [])
    open_pos = ledger.get("open_position")

    c1, c2, c3 = st.columns(3)
    c1.metric("Capital simulado", f"${cap:,.2f}")
    c2.metric("Trades cerrados", len(trades))
    c3.metric("Posicion abierta", "SÍ" if open_pos else "NO")

    if open_pos:
        st.subheader("Posición abierta")
        op = pd.DataFrame([{
            "Lado": open_pos["side"].upper(),
            "Entrada": f"${open_pos['entry_price']:,.2f}",
            "SL": f"${open_pos['sl']:,.2f}",
            "TP": f"${open_pos['tp']:,.2f}",
            "Qty BTC": open_pos["qty"],
            "PNL no realizado": f"${open_pos.get('unrealized_pnl', 0):,.2f}",
            "Apertura": open_pos["open_date"],
        }])
        st.dataframe(op, use_container_width=True)

    if trades:
        st.subheader(f"Trades cerrados ({len(trades)})")
        df_t = pd.DataFrame(trades)
        df_t = df_t.rename(columns={
            "side": "Lado", "entry_price": "Entrada", "exit_price": "Salida",
            "qty": "Qty BTC", "pnl": "PNL $", "pnl_pct": "PNL %",
            "exit_reason": "Razón", "open_date": "Apertura", "close_date": "Cierre",
            "signal_score": "Score señal",
        })
        for col in ["Entrada", "Salida", "PNL $"]:
            if col in df_t.columns:
                df_t[col] = df_t[col].apply(lambda v: f"${v:,.2f}" if isinstance(v, (int, float)) else v)
        if "PNL %" in df_t.columns:
            df_t["PNL %"] = df_t["PNL %"].apply(lambda v: f"{v:,.2f}%" if isinstance(v, (int, float)) else v)
        st.dataframe(df_t[["Lado", "Entrada", "Salida", "Qty BTC", "PNL $", "PNL %", "Razón", "Apertura", "Cierre"]],
                     use_container_width=True)

        # equity acumulada de trades (estilo freqtrade profit curve)
        eq_t = [50.0]
        for t in trades:
            eq_t.append(eq_t[-1] + t.get("pnl", 0.0))
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(y=eq_t, mode="lines", name="Equity (trades)"))
        fig_eq.update_layout(yaxis_title="Capital ($)", xaxis_title="Trade #",
                             title="Equity Curve del Paper Broker")
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("Aún no hay trades cerrados. Ejecuta `python -m live.ledger_bot --action open` "
                 "para abrir la primera posición mensual.")
