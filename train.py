"""Train a reproducible demonstration player-churn classifier on synthetic data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
FEATURES = ["player_level", "weekly_sessions", "avg_session_minutes", "days_since_last_login", "total_playtime_hours", "purchases_last_30d"]

def generate_data(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    level = rng.integers(1, 101, n)
    sessions = rng.poisson(4, n)
    minutes = np.clip(rng.normal(42, 18, n), 2, 180)
    inactive = np.clip(rng.exponential(5, n).astype(int), 0, 30)
    hours = np.clip(level * rng.uniform(1, 5, n) + rng.normal(0, 20, n), 0, None)
    purchases = rng.poisson(0.8, n)
    # Synthetic label: probabilistic simulation, NOT a measured real-world outcome.
    logit = -2.7 + 0.15 * inactive - 0.28 * sessions - 0.008 * minutes - 0.01 * level - 0.22 * purchases
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame(dict(player_level=level, weekly_sessions=sessions, avg_session_minutes=np.round(minutes, 1), days_since_last_login=inactive, total_playtime_hours=np.round(hours, 1), purchases_last_30d=purchases, churn=churn))

def main():
    df = generate_data()
    df.to_csv(ROOT / "synthetic_players.csv", index=False)
    X, y = df[FEATURES], df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    preprocess = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), FEATURES)])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=180, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=2),
    }
    # Select using cross-validation on TRAINING ONLY; holdout test is used once.
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_scores = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        result = cross_validate(pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        cv_scores[name] = float(np.mean(result["test_score"]))
    winner = max(cv_scores, key=cv_scores.get)
    pipeline = Pipeline([("preprocess", preprocess), ("model", models[winner])])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "dataset": "SYNTHETIC DEMONSTRATION DATA — not EA data",
        "rows": len(df), "churn_rate": float(y.mean()),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "cv_average_precision": cv_scores, "selected_model": winner,
        "holdout": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        },
    }
    joblib.dump(pipeline, ROOT / "churn_model.joblib")
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
