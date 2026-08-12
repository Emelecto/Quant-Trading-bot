# DeepFin Trading Bot — Ensemble de Modelos de IA

Bot de trading algorítmico de **investigación aplicada** (paper trading, $0) que
combina modelos de naturaleza distinta mediante **ensemble learning**:

- **Regresión Lineal** (estadístico, interpretable)
- **XGBoost** (machine learning, no lineal)
- **Monte Carlo** (capa probabilística / riesgo)

con extras opcionales: **LSTM**, **CatBoost/LightGBM**, filtro de régimen **ADX**,
**GARCH** (volatilidad adaptativa) y **Stacking** meta-modelo.

> ⚠️ **Proyecto educativo/investigativo.** No garantiza rentabilidad. No usar con
> capital real sin validación exhaustiva y cumplimiento regulatorio.

## Stack
Python 3.11 · Pandas · NumPy · Scikit-learn · XGBoost · ccxt · FastAPI · Streamlit · Docker

## Instalación
```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows (git-bash)
pip install -r requirements.txt
```

## Estructura
```
config/      settings.yaml (activos, riesgo, ensemble, backtest)
data/        fetch_data.py (descarga OHLCV Binance -> raw/processed)
features/    indicators.py, build_features.py
models/      base.py (interfaz) + modelos individuales
ensemble/    voting.py, weighted.py, stacking.py
risk/        risk_management.py (SL=2xATR, TP 1:3, ADX, posición 1%)
backtest/    engine.py (walk-forward), metrics.py (Sharpe, Sortino, MDD...)
api/         FastAPI
dashboard/   Streamlit
tests/       pruebas unitarias (datos sintéticos, sin red)
```

## Uso rápido
```bash
# 1. Descargar datos (BTC, ETH diario, 3 años)
python -m data.fetch_data

# 2. Correr tests unitarios
python -m pytest tests/ -q

# 3. Backtest walk-forward (ver backtest/run.py)
python -m backtest.run
```

## Validación (anti-overfitting)
Todo se evalúa con **walk-forward**: se re-entrena y re-pondera el ensemble en
cada ventana. Benchmark: Buy & Hold + modelo aleatorio. Semillas fijas para
reproducibilidad.

## Cronograma (10-12 semanas)
| Hito | Sem | Entregable |
|---|---|---|
| Pipeline + EDA | 1-3 | datos + features |
| Modelos base | 4-6 | LR, XGBoost, Monte Carlo validados |
| Riesgo + extras | 7-8 | módulo risk, 2º árbol |
| Ensemble + backtest | 9-10 | comparación modelos vs ensemble |
| Dashboard | 11 | Streamlit + FastAPI |
| Doc + presentación | 12 | README, informe, deck |
