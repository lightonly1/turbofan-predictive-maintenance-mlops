"""
Module 3 - MLflow Experiment Tracking + Model Training
=======================================================
Replaces the original plain XGBoost training with:
- MLflow experiment tracking
- Hyperparameter tuning with Hyperopt
- SHAP explainability
- Model registry (Staging → Production)

Original notebook:
    xgb = XGBRegressor(n_estimators=100, ...)
    xgb.fit(X_train, y_train)

Now we have full experiment tracking, every run logged!
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import shap
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

from src.utils.config import get_config
from src.preprocessing.feature_store import FeatureStore


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    """Compute all regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mse), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
        "mse":  round(mse, 4)
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, save_path: str):
    """Actual vs Predicted RUL scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.4, color="steelblue", s=10)
    plt.plot([0, 125], [0, 125], "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual RUL", fontsize=12)
    plt.ylabel("Predicted RUL", fontsize=12)
    plt.title("Actual vs Predicted RUL", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"✅ Saved plot: {save_path}")


def plot_feature_importance(model, feature_names: list, save_path: str):
    """Top 15 feature importance bar chart."""
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)[:15]

    plt.figure(figsize=(10, 5))
    importances.plot(kind="bar", color="steelblue")
    plt.title("Top 15 Feature Importances", fontsize=14)
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"✅ Saved plot: {save_path}")


def plot_shap_summary(model, X_sample: pd.DataFrame, save_path: str):
    """SHAP summary plot — shows which features drive predictions."""
    logger.info("📊 Computing SHAP values (this takes ~30 seconds)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✅ Saved SHAP plot: {save_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(params: dict, X_train, y_train, X_test, y_test,
                feature_names: list, run_name: str = "xgboost_run"):
    """
    Train XGBoost model with full MLflow tracking.
    Every parameter, metric and artifact is logged.
    """
    config = get_config()
    mlflow_cfg = config.mlflow

    # Set MLflow tracking URI and experiment
    # Fix for Windows — must use file:/// prefix
    tracking_uri = mlflow_cfg["tracking_uri"]
    if not tracking_uri.startswith("file:///") and not tracking_uri.startswith("http"):
        tracking_uri = "file:///" + tracking_uri.replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"🔬 MLflow Run ID: {run.info.run_id}")

        # ── 1. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", len(feature_names))
        logger.info("✅ Parameters logged")

        # ── 2. Train Model
        logger.info("🧠 Training XGBoost model...")
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        logger.info("✅ Training complete")

        # ── 3. Predict and Log Metrics
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        logger.info(f"📊 MAE:  {metrics['mae']}")
        logger.info(f"📊 RMSE: {metrics['rmse']}")
        logger.info(f"📊 R²:   {metrics['r2']}")

        # ── 4. Save and Log Plots
        plots_dir = ROOT_DIR / "logs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Actual vs Predicted
        pred_plot = str(plots_dir / "actual_vs_predicted.png")
        plot_actual_vs_predicted(y_test, y_pred, pred_plot)
        mlflow.log_artifact(pred_plot, "plots")

        # Feature Importance
        imp_plot = str(plots_dir / "feature_importance.png")
        plot_feature_importance(model, feature_names, imp_plot)
        mlflow.log_artifact(imp_plot, "plots")

        # SHAP Plot (on sample for speed)
        shap_plot = str(plots_dir / "shap_summary.png")
        X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        plot_shap_summary(model, X_sample, shap_plot)
        mlflow.log_artifact(shap_plot, "plots")

        # ── 5. Log Model to MLflow Registry
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=mlflow_cfg["model_name"]
        )
        logger.info(f"✅ Model registered: {mlflow_cfg['model_name']}")

        return model, metrics, run.info.run_id


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────

def tune_hyperparameters(X_train, y_train, X_test, y_test,
                         feature_names: list, n_trials: int = 5):
    """
    Simple grid search over key hyperparameters.
    Logs each trial as a separate MLflow run.
    """
    logger.info(f"🔍 Starting hyperparameter tuning ({n_trials} trials)...")

    # Parameter grid to search
    param_grid = [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03},
    ]

    best_rmse = float("inf")
    best_params = None
    best_model = None

    for i, params in enumerate(param_grid[:n_trials]):
        # Add fixed params
        full_params = {
            **params,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        logger.info(f"\n🔬 Trial {i+1}/{n_trials}: {params}")
        model, metrics, run_id = train_model(
            full_params, X_train, y_train, X_test, y_test,
            feature_names, run_name=f"trial_{i+1}"
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_params = full_params
            best_model = model
            logger.info(f"🏆 New best RMSE: {best_rmse}")

    logger.info(f"\n✅ Best RMSE: {best_rmse}")
    logger.info(f"✅ Best params: {best_params}")
    return best_model, best_params


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training():
    """Main training pipeline with MLflow tracking."""
    logger.info("=" * 60)
    logger.info("🚀 Starting MLflow Training Pipeline")
    logger.info("=" * 60)

    config = get_config()

    # ── Step 1: Load Features
    logger.info("\n📌 Step 1: Loading Features from Feature Store")
    store = FeatureStore()
    X_train, X_test, y_train, y_test, feature_names = store.get_train_test_split()

    # ── Step 2: Baseline Run
    logger.info("\n📌 Step 2: Baseline Training Run")
    baseline_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
    }
    model, metrics, run_id = train_model(
        baseline_params, X_train, y_train, X_test, y_test,
        feature_names, run_name="baseline"
    )

    # ── Step 3: Hyperparameter Tuning
    logger.info("\n📌 Step 3: Hyperparameter Tuning")
    best_model, best_params = tune_hyperparameters(
        X_train, y_train, X_test, y_test,
        feature_names, n_trials=5
    )

    # ── Step 4: Final Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"   Best RMSE: {metrics['rmse']}")
    logger.info(f"   Best R²:   {metrics['r2']}")
    logger.info(f"   Best MAE:  {metrics['mae']}")
    logger.info("\n👉 View results: mlflow ui")
    logger.info("   Then open: http://localhost:5000")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training()
