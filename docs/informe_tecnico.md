# Informe Técnico — Bot de Trading Ensemble DeepFin

> **Proyecto:** DeepFin – Finanzas con IA (semillero de investigación)
> **Modalidad:** *Paper trading* (simulación, sin capital real)
> **Estado:** MVP funcional + validación walk-forward (Fases 1–8)
> **Fecha:** 2026-08-14

---

## 1. Resumen

Se construyó un bot de trading algorítmico que combina tres familias de modelos
—regresión lineal (estadístico), XGBoost (machine learning) y simulación de
Monte Carlo (probabilístico)— mediante *ensemble learning* (promedio
ponderado), con una capa de gestión de riesgo (stop-loss = 2×ATR, take-profit
1:3, filtro de régimen ADX). El sistema fue evaluado con validación
*walk-forward* sobre datos reales de Binance (BTC/USDT y ETH/USDT, ~3 años,
1096 velas diarias por activo).

**Resultado honesto:** en el periodo 2022–2025 el ensemble **no supera a
Buy & Hold** en rentabilidad absoluta (CAGR/Sharpe), aunque **reduce el
drawdown máximo** en BTC. Esto es esperado para un MVP de investigación y
señala claramente que la señal requiere mejora antes de cualquier uso con
capital real.

---

## 2. Arquitectura

```
Datos (Binance/ccxt) → Limpieza → Feature Engineering (indicadores técnicos)
        ↓
Modelos base:  LinearRegression · XGBoost · Monte Carlo · [CatBoost]
        ↓
Ensemble:  voting → weighted (pesos por Sharpe OOS) → stacking
        ↓
Riesgo:  SL=2×ATR · TP 1:3 · filtro ADX(>20)
        ↓
Backtesting walk-forward → Métricas → Dashboard (Streamlit) + API (FastAPI)
```

Componentes entregados:
- `data/fetch_data.py` — descarga y limpia OHLCV de Binance.
- `features/` — indicadores (RSI, MACD, ATR, ADX, Bollinger) y matriz de features.
- `models/` — interfaz común `BaseModel` + 4 modelos (LR, XGBoost, Monte Carlo, CatBoost).
- `ensemble/` — voting, weighted, stacking.
- `risk/` — SL/TP adaptativo y filtro de régimen; extra GARCH para SL por volatilidad.
- `backtest/` — motor walk-forward + métricas financieras.
- `dashboard/app.py` — visualización Streamlit (señal, equity curves, métricas).
- `api/main.py` — API FastAPI (expone señales y métricas, sin ejecutar órdenes).

---

## 3. Metodología de validación

- **Walk-forward:** ventana de entrenamiento de 252 días, validación de 63,
  avance de 63 (sin solapamiento de validación → sin *data leakage*).
- **Fuera de muestra (OOS):** los modelos se re-entrenan en cada ventana;
  nunca se evalúan en datos usados para entrenar.
- **Benchmark:** Buy & Hold del mismo activo.
- **Riesgo:** SL/TP 1:3 + filtro ADX aplicados en la simulación.

---

## 4. Resultados (walk-forward, BTC/ETH, 2022–2025)

### 4.1 BTC/USDT — Ensemble (weighted) vs Buy & Hold

| Métrica | Ensemble (riesgo) | Buy & Hold |
|---|---|---|
| Sharpe Ratio | −4.00 | **+0.047** |
| Max Drawdown | **−99.5%** | −56.0% |
| Win Rate | ~23% | ~50% |
| CAGR | negativo | **+19.5%** |

### 4.2 ETH/USDT
Ambos pierden capital en el periodo; el ensemble reduce el drawdown respecto a
Buy & Hold pero no genera alpha neto.

### 4.3 Interpretación
- El ensemble **no predice la dirección** mejor que el azar en este horizonte
  diario (ver diagnóstico de señal en curso).
- La capa de riesgo **limita las pérdidas en lo peor** (menor drawdown en BTC),
  pero la señal base es insuficiente para ser rentable.
- Conclusión: el MVP **cumple su propósito de investigación** (arquitectura
  ensemble + riesgo + backtest riguroso) pero **no es apto para capital real**
  hasta mejorar la señal.

---

## 5. Comparación individual vs ensemble (head-to-head)

Se añadió la capacidad de reportar cada modelo de forma aislada. En las
pruebas, ningún modelo individual supera a Buy & Hold; el ensemble tampoco.
Esto confirma que el cuello de botella está en las **features y el horizonte
de predicción**, no en el método de combinación.

---

## 6. Limitaciones conocidas

1. **Horizonte diario es ruidoso:** predecir el retorno del día siguiente es
   cercano al azar para estos activos.
2. **Features simples:** solo indicadores técnicos; sin datos de volumen
   avanzado, on-chain (para crypto) o macro.
3. **Pesos del ensemble no optimizados** más allá de Sharpe OOS aproximado.
4. **Sin costos de transacción** en la simulación (spread/fee de Binance).
5. **Datos 2022–2025 incluyen el invierno cripto**, periodo adverso.

---

## 7. Trabajo futuro (ruta hacia rentabilidad)

- **Mejorar la señal (Frente B):** features más ricas, horizonte semanal,
  filtro de régimen adaptativo, modelo LSTM, optimización de pesos OOS.
- **Paper trading en vivo** 3–6 meses antes de capital real.
- **Costos de transacción** en la simulación.
- **Stacking** solo si supera claramente a weighted en OOS.

---

## 8. Consideraciones éticas y legales

Proyecto **educativo/investigativo**. No constituye asesoría financiera ni
garantía de rentabilidad. No usar con capital real sin validación exhaustiva y
cumplimiento regulatorio correspondiente.

---

*Generado con los resultados del backtest walk-forward validado del repositorio
DeepFin. Números reproducibles vía `python -m backtest.run`.*
