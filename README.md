# DeepFin — Bot de Trading Ensemble (Paper Trading)

> Semillero **DeepFin – Finanzas con IA**. Proyecto de investigación aplicada:
> un bot de trading algorítmico que combina modelos estadísticos y de machine
> learning mediante *ensemble learning*, validado con *backtesting walk-forward*.
> **Modalidad paper trading (simulación), sin capital real.**

---

## ⚠️ Aviso

Proyecto **educativo/investigativo**. No constituye asesoría financiera ni
garantía de rentabilidad. No usar con capital real sin validación exhaustiva y
cumplimiento regulatorio.

---

## Arquitectura

```
Datos (Binance/ccxt) → Limpieza → Features (indicadores técnicos)
        ↓
Modelos:  LinearRegression · XGBoost · Monte Carlo · [CatBoost]
        ↓
Ensemble:  voting → weighted → accuracy-weighted → stacking
        ↓
Riesgo:  SL=2×ATR · TP 1:3 · filtro ADX(>20) · [GARCH SL]
        ↓
Backtesting walk-forward → Métricas → Dashboard (Streamlit) + API (FastAPI)
```

## Modelos base
| Modelo | Tipo | Señal |
|---|---|---|
| Linear Regression | Estadístico | score en [-1, +1] |
| XGBoost | Machine Learning | 2·P(suba) − 1 |
| Monte Carlo | Probabilístico | % trayectorias alcistas (capa de riesgo) |
| CatBoost | ML (2º árbol) | 2·P(suba) − 1 |

## Ensemble
- `voting` — mayoría de direcciones.
- `weighted` — promedio por Sharpe OOS de la ventana train.
- `accuracy` — promedio por accuracy direccional OOS (evita diluir la señal útil con modelos de ruido).
- `stacking` — meta-modelo logístico (solo si supera a `weighted`).

## Resultados (walk-forward, BTC/ETH 2022–2025)
El MVP **no supera a Buy & Hold** en rentabilidad absoluta, pero **reduce el
drawdown** en BTC. Ver `docs/informe_tecnico.md` para el análisis completo y
las métricas reproducibles. El hallazgo clave: XGBoost predice con ~73%
accuracy direccional, pero el ensemble equal-weight lo diluía; el método
`accuracy-weighted` lo corrige (en evaluación).

---

## Cómo ejecutar

```bash
# 1. Crear y activar entorno (fuera de OneDrive recomendado)
python -m venv C:/venvs/deepfin
C:/venvs/deepfin/Scripts/activate
pip install -r requirements.txt

# 2. Descargar datos reales (Binance)
python -m data.fetch_data

# 3. Backtest walk-forward (modelos individuales vs ensemble vs Buy&Hold)
python -m backtest.run
python -m backtest.run --catboost      # incluye CatBoost
python -m backtest.run --garch         # SL por volatilidad GARCH

# 4. Pruebas
python -m pytest tests/ -q

# 5. Dashboard (Streamlit)
streamlit run dashboard/app.py --server.port 8501

# 6. API (FastAPI)
uvicorn api.main:app --reload
```

## Estructura
```
config/        settings.yaml
data/          fetch_data.py (descarga/limpia OHLCV)
features/      indicadores + matriz de features
models/        base + LR, XGBoost, MonteCarlo, CatBoost
ensemble/      voting, weighted, accuracy, stacking
risk/          SL/TP, filtro ADX, GARCH
backtest/      motor walk-forward + métricas
dashboard/     app.py (Streamlit)
api/           main.py (FastAPI)
tests/         pruebas unitarias
docs/          informe_tecnico.md
```

## Stack
Python 3.11 · pandas · numpy · scikit-learn · xgboost · catboost · ccxt ·
statsmodels · streamlit · fastapi · plotly.
