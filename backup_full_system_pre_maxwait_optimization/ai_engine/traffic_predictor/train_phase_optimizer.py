"""
Machine Learning Phasing Strategy & Green Duration Optimizer Trainer
Trains Random Forest Classifier (Phasing Policy) & Gradient Boosting Regressor (Green Duration)
General Sir John Kotelawala Defence University (KDU) - IT 3182 Essentials of AI
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score, mean_absolute_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FEATURE_COLS = [
    "sin_time", "cos_time", "is_rush_hour",
    "flow_N", "flow_S", "flow_E", "flow_W",
    "pcu_N_L0", "pcu_N_L1",
    "pcu_S_L0", "pcu_S_L1",
    "pcu_E_L0", "pcu_E_L1",
    "pcu_W_L0", "pcu_W_L1",
    "tot_pcu_N", "tot_pcu_S", "tot_pcu_E", "tot_pcu_W",
    "max_wait_sec", "asym_ratio"
]

POLICY_MAP = {
    0: "BALANCED_PHASE",
    1: "N_EXCLUSIVE_GREEN",
    2: "S_EXCLUSIVE_GREEN",
    3: "E_EXCLUSIVE_GREEN",
    4: "W_EXCLUSIVE_GREEN"
}

def train_phase_optimizer():
    csv_path = os.path.join(PROJECT_ROOT, "data", "datasets", "phase_optimization_dataset.csv")
    if not os.path.exists(csv_path):
        print("Dataset not found. Generating dataset first...")
        from ai_engine.traffic_predictor.dataset_phase_generator import generate_phase_dataset
        df = generate_phase_dataset(output_csv=csv_path)
    else:
        df = pd.read_csv(csv_path)

    print("=" * 65)
    print(f"TRAINING ML PHASING & GREEN DURATION OPTIMIZER ({len(df)} samples)")
    print("=" * 65)

    X = df[FEATURE_COLS]
    y_cls = df["optimal_policy_code"]
    y_reg = df["optimal_green_sec"]

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.20, random_state=42, stratify=y_cls
    )

    # 1. Train Phasing Policy Classifier
    print("\nTraining Random Forest Phasing Policy Classifier...")
    clf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_cls_train)

    y_cls_pred = clf.predict(X_test)
    acc = accuracy_score(y_cls_test, y_cls_pred)
    print(f"Policy Classification Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_cls_test, y_cls_pred, target_names=[POLICY_MAP[i] for i in range(5)]))

    # 2. Train Green Duration Regressor
    print("\nTraining Gradient Boosting Green Duration Regressor...")
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    reg.fit(X_train, y_reg_train)

    y_reg_pred = reg.predict(X_test)
    r2 = r2_score(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    print(f"Green Time Regression R² Score: {r2:.4f}")
    print(f"Green Time RMSE: {rmse:.2f} seconds | MAE: {mae:.2f} seconds")

    # Save Model Bundle
    models_dir = os.path.join(PROJECT_ROOT, "ai_engine", "traffic_predictor", "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    model_bundle_path = os.path.join(models_dir, "ml_phase_optimizer.joblib")

    joblib.dump({
        "classifier": clf,
        "regressor": reg,
        "feature_cols": FEATURE_COLS,
        "policy_map": POLICY_MAP,
        "metrics": {
            "accuracy": acc,
            "r2": r2,
            "rmse": rmse,
            "mae": mae
        }
    }, model_bundle_path)
    print(f"\nTrained Model Bundle saved to: {model_bundle_path}")

    # Generate Evaluation Chart
    plt.figure(figsize=(12, 5))
    
    # Feature Importances
    plt.subplot(1, 2, 1)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[-8:]
    plt.barh(range(len(indices)), importances[indices], color="#2ecc71", align="center")
    plt.yticks(range(len(indices)), [FEATURE_COLS[i] for i in indices])
    plt.title("ML Phasing Policy - Top Feature Importances")
    plt.xlabel("Gini Importance")

    # Regression Fit
    plt.subplot(1, 2, 2)
    plt.scatter(y_reg_test[:200], y_reg_pred[:200], alpha=0.6, color="#3498db", edgecolors="none")
    plt.plot([8, 45], [8, 45], 'r--', lw=2, label="Ideal Fit (y = x)")
    plt.xlabel("Actual Optimal Green (s)")
    plt.ylabel("ML Predicted Green (s)")
    plt.title(f"Green Duration Prediction (R² = {r2:.4f}, MAE = {mae:.2f}s)")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(models_dir, "ml_phase_evaluation_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Evaluation Plot saved to: {plot_path}")

if __name__ == "__main__":
    train_phase_optimizer()
