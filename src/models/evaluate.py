"""
Model Evaluation — Metrics, SHAP feature importance
"""

import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR = Path("models")


def load_models():
    return {
        "disorder":  joblib.load(MODELS_DIR / "model_disorder.pkl"),
        "treatment": joblib.load(MODELS_DIR / "model_treatment.pkl"),
        "scaler":    joblib.load(MODELS_DIR / "scaler.pkl"),
        "features":  joblib.load(MODELS_DIR / "feature_names.pkl"),
    }


def shap_summary(model, X, feature_names: list, save_path: str = None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
