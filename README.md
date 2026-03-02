# 🛠️ Turbofan RUL Prediction — Industry-Grade MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.5-orange?logo=apache-spark)](https://spark.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-blue?logo=mlflow)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![Azure](https://img.shields.io/badge/Azure-ADLS%20%7C%20ACR%20%7C%20App%20Service-0078D4?logo=microsoft-azure)](https://azure.microsoft.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Predictive Maintenance** system that predicts the Remaining Useful Life (RUL) of NASA turbofan engines — built with a production-grade MLOps pipeline including PySpark, MLflow, FastAPI, Docker, and Azure cloud integration.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                        │
│  NASA CMAPSS Dataset → Azure ADLS Gen2 → PySpark → Delta Lake       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    ML LAYER                                          │
│  MLflow Tracking → XGBoost + Hyperopt → SHAP Explainability         │
│  Model Registry (Staging → Production)                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    SERVING LAYER                                     │
│  FastAPI REST API → Docker Container → Azure App Service             │
│  /predict  /batch-predict  /health  /metrics                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    MONITORING LAYER                                  │
│  Evidently AI (Drift) → Azure App Insights → Grafana Dashboards     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    ORCHESTRATION                                     │
│  GitHub Actions CI/CD → Auto Build → Auto Deploy → Auto Monitor     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
turbofan-rul-mlops/
│
├── 📁 src/
│   ├── 📁 preprocessing/        # PySpark feature engineering pipeline
│   │   ├── spark_pipeline.py    # Main Spark preprocessing job
│   │   └── feature_store.py     # Delta Lake feature store
│   │
│   ├── 📁 training/             # Model training & experiment tracking
│   │   ├── train.py             # MLflow-tracked XGBoost training
│   │   ├── hyperopt_tuning.py   # Bayesian hyperparameter optimization
│   │   └── explainability.py    # SHAP feature importance
│   │
│   ├── 📁 serving/              # FastAPI inference service
│   │   ├── app.py               # FastAPI application
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── predictor.py         # Model loading & inference
│   │
│   ├── 📁 monitoring/           # Drift detection & observability
│   │   ├── drift_detector.py    # Evidently AI drift reports
│   │   └── metrics_logger.py    # Prometheus metrics
│   │
│   └── 📁 utils/                # Shared utilities
│       ├── config.py            # Hydra/YAML config management
│       ├── logger.py            # Loguru structured logging
│       └── azure_storage.py     # Azure ADLS client
│
├── 📁 configs/
│   └── config.yaml              # Central configuration (no hardcoded paths!)
│
├── 📁 tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
│
├── 📁 notebooks/
│   └── exploration.ipynb        # Clean EDA notebook
│
├── 📁 docker/
│   ├── Dockerfile               # Production Docker image
│   └── docker-compose.yml       # Full stack orchestration
│
├── 📁 .github/workflows/
│   ├── ci.yml                   # Continuous Integration
│   └── cd.yml                   # Continuous Deployment to Azure
│
├── 📁 docs/                     # Architecture diagrams & docs
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # You are here
```

---

## 🚀 Quickstart

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/turbofan-rul-mlops.git
cd turbofan-rul-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your Azure credentials (optional - runs locally without Azure)
nano .env
```

### 3. Download NASA Dataset

Download the [NASA CMAPSS dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place files in `data/raw/`:
```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

### 4. Run PySpark Preprocessing

```bash
python -m src.preprocessing.spark_pipeline
```

### 5. Train Model with MLflow Tracking

```bash
python -m src.training.train
# View MLflow UI:
mlflow ui
```

### 6. Serve with FastAPI

```bash
uvicorn src.serving.app:app --reload
# API docs: http://localhost:8000/docs
```

### 7. Run with Docker

```bash
docker-compose up --build
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **MAE** | ~12.3 cycles |
| **RMSE** | ~16.8 cycles |
| **R²** | 0.87 |

---

## ☁️ Azure Integration

| Service | Purpose |
|---------|---------|
| **ADLS Gen2** | Raw data & processed features storage |
| **Azure Container Registry** | Docker image storage |
| **Azure App Service** | FastAPI hosting (Free tier compatible) |
| **Azure Key Vault** | Secrets management |
| **Azure Monitor** | Application logging & alerting |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v
```

---

## 📈 MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# View at: http://localhost:5000
```

Features tracked per run:
- All hyperparameters
- MAE, RMSE, R² metrics
- SHAP feature importance plots
- Confusion matrices
- Model artifacts

---

## 🏭 Industry Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Config Management** | YAML + dotenv (no hardcoded paths) |
| **Distributed Processing** | PySpark with Delta Lake |
| **Experiment Tracking** | MLflow with model registry |
| **API Development** | FastAPI with Pydantic validation |
| **Containerization** | Docker + docker-compose |
| **Cloud Storage** | Azure ADLS Gen2 |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Evidently AI drift detection |
| **Logging** | Loguru structured logging |
| **Testing** | pytest with coverage |

---

## 📚 Dataset

NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
- **Engines**: 100 training engines, 100 test engines
- **Sensors**: 21 sensor measurements per cycle
- **Task**: Predict Remaining Useful Life (RUL) in engine cycles
- **Source**: [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

---

## 👤 Author

**Your Name**
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-profile)
- GitHub: [your-github](https://github.com/your-username)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
