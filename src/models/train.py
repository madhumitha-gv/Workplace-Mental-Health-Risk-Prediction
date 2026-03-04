"""
Model Training Pipeline
Trains XGBoost (disorder) and Gradient Boosting (treatment) classifiers
on the top-10 most correlated features per target.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier

from src.data.preprocessing import preprocess_data, create_composite_indices

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_DISORDER  = "Do you currently have a mental health disorder?"
TARGET_TREATMENT = "Have you ever sought treatment for a mental health issue from a mental health professional?"


def get_top_corr_features(df: pd.DataFrame, target: str, n: int = 10) -> list:
    """Return top-n features most correlated with the target column."""
    numeric_df = df.select_dtypes(include=[np.number])
    if target not in numeric_df.columns:
        return []
    corr = numeric_df.corr()[target].abs().drop(target).sort_values(ascending=False)
    return corr.head(n).index.tolist()


def train(data_path: str = "data/raw/mental_health.csv"):
    df_raw = pd.read_csv(data_path)
    df_clean = preprocess_data(df_raw)
    df_features = create_composite_indices(df_clean)

    results = {}

    for target, model_name in [(TARGET_DISORDER, "disorder"), (TARGET_TREATMENT, "treatment")]:
        if target not in df_features.columns:
            print(f"[WARN] Target column not found: {target}")
            continue

        top_features = get_top_corr_features(df_features, target)
        print(f"\n[{model_name}] Top 10 features: {top_features}")

        X = df_features[top_features].fillna(0)
        y = df_features[target].fillna(0).round().astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                 use_label_encoder=False, eval_metric="logloss",
                                 random_state=42) if model_name == "disorder" \
            else GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, random_state=42)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        f1 = f1_score(y_test, preds, average="weighted")
        acc = accuracy_score(y_test, preds)

        print(f"[{model_name}] Accuracy: {acc:.3f} | F1: {f1:.3f}")
        print(classification_report(y_test, preds))

        joblib.dump(clf, MODELS_DIR / f"model_{model_name}.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(top_features, MODELS_DIR / "feature_names.pkl")

        results[model_name] = {"f1": f1, "accuracy": acc, "features": top_features}

    return results


if __name__ == "__main__":
    train()
