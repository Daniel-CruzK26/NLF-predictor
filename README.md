# 🏈 NFL Gridiron AI Predictor — Machine Learning Spread & Value Betting Engine

Un sistema predictivo avanzado de Machine Learning y analítica cuantitativa para modelar el **Spread (margen de puntos)**, la **probabilidad de victoria** y detectar apuestas con valor esperado positivo (**+EV Picks**) en la NFL.

---

## 🌟 Características Principales

1. **Ingeniería de Características Cuantitativas**:
   - **EPA por jugada** (Pase Ofensivo vs Defensivo, Carrera Ofensiva vs Defensiva, Tasa de Éxito).
   - **Ajuste por Rival (*Opponent-Adjusted EPA*)**: Regresión Ridge dinámica para aislar la verdadera eficiencia de cada unidad.
   - **Medias Móviles Exponenciales (*EWMA*)**: Ventanas con decaimiento ($\alpha = 0.15$) y técnica anti-fuga de datos (`.shift(1)`).
   - **Métricas Avanzadas de QB**: EPA por dropback, CPOE (Completion Percentage Over Expected) y Composite Index con encogimiento Bayesiano hacia nivel de reemplazo.
   - **Factores Situacionales**: Diferencial de descanso, ventaja de localía (HFA), domo/abierto, velocidad del viento y temperatura.

2. **Modelado y Validación Temporal Estricta (*Time-Series Split*)**:
   - **LightGBM Regressor** (MAE de **9.719 pts**, **67.14% SU**, **55.57% ATS** superando el umbral de rentabilidad del 52.4%).
   - **XGBoost Regressor** (MAE de **9.843 pts**, **67.02% SU**, **53.87% ATS**).
   - **Baseline Elo Dinámico** (con multiplicador de Margen de Victoria y reversión estacional).
   - **Ensamble Ponderado de Producción** (50% LightGBM + 30% XGBoost + 20% Ridge).

3. **Sincronización en Tiempo Real con ESPN**:
   - Módulo integrado para consultar los *depth charts* oficiales y plantillas de los 32 equipos en tiempo real desde la API de ESPN.

4. **Dashboard Web Interactivo (FastAPI + Glassmorphism Dark UI)**:
   - **Cartelera y Picks +EV**: Nivel de confianza (3★, 2★, 1★), Kelly Criterion fraccional (1/4 Kelly) y análisis profundo de cada partido.
   - **Power Rankings**: 32 equipos ordenados por Power Rating Compuesto.
   - **Model Lab**: Comparativa de error out-of-fold e importancia de variables.
   - **Gestor de Plantillas**: Editor visual y sincronizador de QBs titulares.
   - **Simulador Interactivo**: Simulación de cualquier partido con condiciones climáticas personalizadas.

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio y configurar entorno
```bash
git clone https://github.com/<tu-usuario>/NFLpredictor.git
cd NFLpredictor

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Iniciar el Dashboard Web
```bash
python scripts/run_dashboard.py
```
Abre en tu navegador: **`http://127.0.0.1:8000`**

### 3. Ejecutar Pipeline Analítico por CLI
```bash
# Ingesta y agregación de datos
python scripts/01_fetch_and_aggregate.py

# Benchmark de modelo Elo
python scripts/02_run_elo_benchmark.py

# Cálculo de variables EWMA
python scripts/03_compute_ewma_features.py

# Ajuste por Rival y Métricas de QB
python scripts/04_build_advanced_features.py

# Entrenamiento de Modelos de Machine Learning
python scripts/05_train_ml_models.py

# Predicción de la Semana con Detección de Valor (+EV)
python scripts/06_predict_upcoming_week.py --season 2026 --week 1
```

### 4. Ejecutar Suite de Pruebas
```bash
pytest -v
```

---

## 📂 Estructura del Proyecto

```
NFLpredictor/
├── data/                      # Datos procesados y configuraciones de rosters
│   ├── active_starters.json   # QBs titulares activos por equipo
│   ├── team_modifiers.json    # Calibraciones de Power Ratings
│   └── processed/             # Tablas intermedias de entrenamiento y predicción
├── scripts/                   # Scripts CLI ejecutables (Fases 1 a 6 y utilidades)
│   ├── 01_fetch_and_aggregate.py
│   ├── 02_run_elo_benchmark.py
│   ├── 03_compute_ewma_features.py
│   ├── 04_build_advanced_features.py
│   ├── 05_train_ml_models.py
│   ├── 06_predict_upcoming_week.py
│   ├── fetch_espn_rosters.py
│   └── run_dashboard.py
├── src/                       # Código fuente modular
│   ├── config.py              # Parámetros globales y mapeos
│   ├── config_rosters.py      # Gestor dinámico de plantillas
│   ├── data/                  # Ingesta y agregación jugada a jugada
│   ├── features/              # EWMA, Opponent Adjustment, QBs y Clima
│   ├── models/                # Elo, Trainer ML, Evaluador y Motor de Inferencia
│   └── web/                   # Servidor FastAPI, CSS, JS y UI
├── tests/                     # Suite de pruebas unitarias con Pytest
├── requirements.txt           # Dependencias del proyecto
└── README.md
```

---

## 🛡️ Licencia
MIT License — Código abierto para análisis cuantitativo deportivo.
