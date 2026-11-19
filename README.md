# Game Player Churn Prediction — EA-aligned ML portfolio demo

An end-to-end educational ML project: synthetic data generation, preprocessing, stratified train/test split, imbalanced classification, cross-validation, holdout evaluation, model persistence, and a Streamlit prediction interface.

**Important:** All player records and churn labels are synthetic. This is NOT Electronic Arts data. The Streamlit app is a local deployment demonstration, not a hosted production service. Do not claim real-world predictive performance or production deployment.

## Quick start

```bash
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

`train.py` generates `synthetic_players.csv`, `metrics.json`, and `churn_model.joblib` in this folder. `app.py` loads the trained model and offers interactive predictions.

## Evaluation methodology

- Stratified 75/25 train/test split.
- Missing-value imputation and scaling happen inside a scikit-learn Pipeline, avoiding train/test preprocessing leakage.
- Compare Logistic Regression (balanced weights), Random Forest (balanced weights), and XGBoost.
- Select model by 4-fold training-only cross-validation average precision (PR-AUC).
- Evaluate the selected model once on the untouched holdout with accuracy, precision, recall, F1, ROC-AUC and average precision.
- Churn is rare in the synthetic generator; accuracy alone is misleading.

## Next steps for a stronger EA portfolio project

1. Replace synthetic data with a permitted, real player engagement dataset containing longitudinal activity and a trustworthy churn label.
2. Define a prediction cutoff and future churn window; ensure no post-cutoff features leak into training.
3. Use a temporal train/test split when records have timestamps.
4. Tune decision thresholds on validation data; evaluate calibration, drift and fairness where appropriate.
5. Containerize and host the Streamlit app or expose a FastAPI endpoint, then add monitoring.

## Resume guidance

Mention this as a **synthetic-data prototype** until real-world data and deployment are added. Only cite metrics actually obtained by running the training script.
