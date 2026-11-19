"""Streamlit demo of a player churn prediction model."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Player Churn Prediction", page_icon="🎮")
st.title("🎮 Player Churn Prediction")
st.warning("Educational demo trained on SYNTHETIC data, not real EA data. Predictions are not validated for real players.")
if not (ROOT / "churn_model.joblib").exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()
model = joblib.load(ROOT / "churn_model.joblib")
with st.form("player"):
    level = st.slider("Player level", 1, 100, 30)
    sessions = st.slider("Sessions per week", 0, 30, 4)
    duration = st.slider("Average session duration (minutes)", 2, 180, 40)
    inactivity = st.slider("Days since last login", 0, 30, 5)
    hours = st.number_input("Total playtime hours", min_value=0.0, value=100.0)
    purchases = st.slider("Purchases in last 30 days", 0, 20, 0)
    submitted = st.form_submit_button("Predict churn")
if submitted:
    row = pd.DataFrame([dict(player_level=level, weekly_sessions=sessions, avg_session_minutes=duration, days_since_last_login=inactivity, total_playtime_hours=hours, purchases_last_30d=purchases)])
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated churn probability (synthetic demo)", f"{probability:.1%}")
    st.progress(probability)
    st.caption("The 50% threshold is illustrative, not a calibrated operational decision rule.")
with st.expander("Model evaluation"):
    p = ROOT / "metrics.json"
    if p.exists():
        metrics = json.loads(p.read_text(encoding="utf-8"))
        st.write("Selected model:", metrics["selected_model"])
        st.json(metrics["holdout"])
