"""
Machine Learning Traffic Forecasting Model Trainer
Trains Random Forest, Gradient Boosting, and Ridge models on multi-week traffic datasets
with Directional Tidal Flow Asymmetry.
Saves trained model artifacts and generates academic evaluation plots.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

FEATURE_COLS = [
    "hour", "minute", "day_of_week", "is_weekend",
    "sin_hour", "cos_hour", "sin_day", "cos_day",
    "is_morning_rush", "is_evening_rush", "weather_multiplier",
    "lag_pcu_N", "lag_pcu_S", "lag_pcu_E", "lag_pcu_W"
]

REG_TARGET_COLS = [
    "flow_north_vpm", "flow_south_vpm", "flow_east_vpm", "flow_west_vpm",
    "target_green_NS", "target_green_EW"
]

CLS_TARGET_COL = "critical_approach"


def train_and_evaluate(dataset_csv_path: str, save_dir: str):
    print("=" * 60)
    print("TRAINING SCIKIT-LEARN TRAFFIC CONGESTION PREDICTOR")
    print("=" * 60)

    if not os.path.exists(dataset_csv_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_csv_path}")

    df = pd.read_csv(dataset_csv_path)
    print(f"Loaded dataset with {len(df)} samples.")

    X = df[FEATURE_COLS]
    y_reg = df[REG_TARGET_COLS]
    y_cls = df[CLS_TARGET_COL]

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.20, random_state=42, shuffle=True
    )

    print(f"Training samples: {len(X_train)} | Holdout test samples: {len(X_test)}")

    # 1. Train Flow & Timing Regression Models (Random Forest)
    print("\nTraining Random Forest Regressor (Multi-Output)...")
    rf_reg = RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_reg_train)

    y_reg_pred_rf = rf_reg.predict(X_test)
    r2_rf = r2_score(y_reg_test, y_reg_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_rf))
    mae_rf = mean_absolute_error(y_reg_test, y_reg_pred_rf)

    print(f"--> Random Forest Regressor Results: R2 = {r2_rf:.4f} | RMSE = {rmse_rf:.2f} | MAE = {mae_rf:.2f}")

    # Baseline comparison with Ridge
    print("\nTraining Ridge Regression Baseline...")
    ridge_reg = Ridge(alpha=1.0)
    ridge_reg.fit(X_train, y_reg_train)
    y_reg_pred_ridge = ridge_reg.predict(X_test)
    r2_ridge = r2_score(y_reg_test, y_reg_pred_ridge)
    print(f"--> Ridge Baseline Results: R2 = {r2_ridge:.4f}")

    # 2. Train Critical Approach Classifier
    print("\nTraining Critical Approach Classifier...")
    rf_cls = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
    rf_cls.fit(X_train, y_cls_train)

    y_cls_pred = rf_cls.predict(X_test)
    acc = accuracy_score(y_cls_test, y_cls_pred)
    print(f"--> Critical Approach Classifier Accuracy = {acc * 100:.2f}%")

    # 3. Save Model Artifact Bundle
    os.makedirs(save_dir, exist_ok=True)
    model_bundle = {
        "regressor": rf_reg,
        "classifier": rf_cls,
        "feature_cols": FEATURE_COLS,
        "reg_target_cols": REG_TARGET_COLS,
        "cls_target_col": CLS_TARGET_COL,
        "metrics": {
            "r2_score": round(r2_rf, 4),
            "rmse": round(rmse_rf, 2),
            "mae": round(mae_rf, 2),
            "accuracy": round(acc * 100, 2)
        }
    }
    bundle_path = os.path.join(save_dir, "traffic_predictor.joblib")
    joblib.dump(model_bundle, bundle_path)
    print(f"\nTrained Model Bundle saved to: {bundle_path}")

    # 4. Generate Visual Evaluation Plots
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    importances = rf_reg.feature_importances_
    indices = np.argsort(importances)
    plt.title("ML Feature Importance Ranking (Random Forest)", fontsize=11, fontweight="bold")
    plt.barh(range(len(indices)), importances[indices], color="#3498db", align="center")
    plt.yticks(range(len(indices)), [FEATURE_COLS[i] for i in indices])
    plt.xlabel("Relative Importance")

    plt.subplot(1, 2, 2)
    actual_north = y_reg_test["flow_north_vpm"].values[:60]
    pred_north = y_reg_pred_rf[:60, 0]
    plt.plot(actual_north, label="Actual Flow (North v/m)", color="#2ecc71", linewidth=2)
    plt.plot(pred_north, label="Predicted Flow (North v/m)", color="#e74c3c", linestyle="--", linewidth=2)
    plt.title(f"Actual vs Predicted Arrival Flow (R2={r2_rf:.3f})", fontsize=11, fontweight="bold")
    plt.xlabel("Test Sample Index (15-min Intervals)")
    plt.ylabel("Arrival Flow (veh/min)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "model_evaluation_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Evaluation Plot saved to: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    csv_path = os.path.join(PROJECT_ROOT, "data", "datasets", "historical_traffic_data.csv")
    models_dir = os.path.join(PROJECT_ROOT, "ai_engine", "traffic_predictor", "saved_models")
    train_and_evaluate(csv_path, models_dir)
