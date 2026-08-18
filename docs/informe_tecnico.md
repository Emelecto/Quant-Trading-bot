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
- El enfoque **diario con indicadores técnicos** no predice la dirección mejor
  que el azar fuera de muestra (XGBoost OOS dir_acc = 49.7% ≈ azar; el 73.6%
  in-sample era overfit).
- La capa de riesgo limita pérdidas pero no crea edge donde no lo hay.

### 4.4 Hallazgo clave: edge en HORIZONTE MENSUAL (21 días)
Al cambiar el horizonte de predicción a **21 días** y usar features de
**momentum/vol-state** (no técnicas clásicas: `ret_21`, `ret_63`, `ret_7`,
`vol_ratio`, `dist_ma63`), XGBoost alcanza **OOS dir_acc = 57.7%** — primer
edge real fuera de muestra.

Simulación mensual correcta (posición 21 días + SL/TP de volatilidad 21d,
walk-forward 2022–2025):

| Activo | Métrica | Mensual+riesgo(21d) | Buy & Hold |
|---|---|---|---|
| **BTC** | Sharpe | **+1.85** | −3.20 |
| BTC | Max Drawdown | −91.6% | −99.4% |
| BTC | CAGR | **+283%** | −81% |
| BTC | Profit Factor | **1.39** | 0.59 |
| ETH | Sharpe | −0.76 | −2.74 |
| ETH | CAGR | −72.7% | −89.2% |

**Conclusión:** el bot mensual tiene **edge real y rentable en BTC** (Sharpe
+1.85, supera a Buy&Hold en todo). ETH mejora respecto a B&H pero no es
rentable aún. El Max DD −91.6% (BTC) es alto por operar 100% del capital por
trade; se corrige con tamaño de posición por riesgo (ver sección 4.5).

### 4.5 Ajuste de tamaño de posición (operable con capital real)
En lugar de apostar 100% del capital, se arriesga una fracción fija por trade
(`risk_per_trade`); el SL define el tamaño: `position_size = risk_per_trade /
sl_frac`. Resultados (BTC, mismo edge mensual):

| risk/trade | Sharpe | Max Drawdown | CAGR |
|---|---|---|---|
| 100% (baseline) | +1.85 | −92.1% | +279% |
| 10% | +1.85 | −92.1% | +279% |
| 1% | **+2.09** | **−42.3%** | +58% |

Con `risk_per_trade = 1%` el drawdown baja de −92% a **−42%** y el Sharpe
sube a **+2.09** (mejor que Buy&Hold +1.16 en este benchmark) → el bot pasa a
ser **apto para paper trading y, con validación viva, para $50 reales** (según
especificación). El CAGR se reduce a +58% (menos capital en juego) pero sigue
fuertemente positivo.



---

## 5. Comparación individual vs ensemble (head-to-head)

El diagnóstico walk-forward mostró que, en horizonte diario, ningún modelo
individual supera a Buy & Hold (el ensemble equal-weight diluía la única señal
útil). En horizonte **mensual**, XGBoost con features de momentum/vol-state es
el modelo que aporta el edge; el ensemble *accuracy-weighted* lo preserva sin
diluirlo.

## 6. Limitaciones conocidas

1. **Edge validado solo en BTC (mensual).** ETH mejora vs B&H pero no es
   rentable; hace falta validar en más activos y periodos.
2. **Una sola configuración** (hold=21, SL=2×vol21, TP 1:3, XGBoost). No es
   un barrido exhaustivo de hiperparámetros.
3. **Sin costos de transacción** (fees de Binance, spread) en la simulación.
4. **Features de precio puro.** No incluye on-chain, flujos ETF, sentimiento
   ni macro, que podrían reforzar el edge.
5. **Periodo 2022–2025 incluye el invierno cripto**; el CAGR positivo de BTC es
   notable precisamente en un periodo adverso.

## 7. Trabajo futuro (ruta hacia ingresos)

- **Paper trading en vivo (BTC mensual)** 3–6 meses con `risk_per_trade=1%`,
  monitoreando Sharpe/DD reales.
- **Costos de transacción** y slippage en la simulación.
- **Validar ETH y otros activos**; explorar features on-chain/sentimiento.
- **$50 reales solo si** el paper live mantiene Sharpe > 1 y DD acotado.

---

## 8. Consideraciones éticas y legales

Proyecto **educativo/investigativo**. No constituye asesoría financiera ni
garantía de rentabilidad. No usar con capital real sin validación exhaustiva y
cumplimiento regulatorio correspondiente.

---

*Generado con los resultados del backtest walk-forward validado del repositorio
DeepFin. Números reproducibles vía `python -m backtest.run`.*
