import streamlit as st
import pandas as pd
import joblib
import gzip
import numpy as np
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Cyber Threat Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- LOAD MODEL (SAFE) ----------------
@st.cache_resource
def load_model():
    if os.path.exists("unsw_rf_model.pkl.gz"):
        with gzip.open("unsw_rf_model.pkl.gz", "rb") as f:
            return joblib.load(f)
    elif os.path.exists("unsw_rf_model.pkl"):
        return joblib.load("unsw_rf_model.pkl")
    else:
        st.error("❌ Model file not found. Please check repository.")
        st.stop()

@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")

@st.cache_data
def load_data():
    df = pd.read_parquet("dataset/UNSW_NB15_testing.parquet")
    return df.sample(5000, random_state=42)

# ---------------- LOAD ASSETS ----------------
model = load_model()
label_encoder = load_label_encoder()
df = load_data()

# ---------------- UI ----------------
st.title("🛡️ AI-Based Cyber Threat Prediction Dashboard")
st.markdown("Autonomous Cyber Defense (Simulation)")

# ---------------- PREP ----------------
X = df.drop(columns=["attack_cat"])
X = X.replace("-", np.nan)

# ---------------- PREDICT ----------------
pred_encoded = model.predict(X)
pred_labels = label_encoder.inverse_transform(pred_encoded)
df["Predicted_Attack"] = pred_labels

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Detected Attacks", (df["Predicted_Attack"] != "Normal").sum())
col3.metric("Normal Traffic", (df["Predicted_Attack"] == "Normal").sum())

# ---------------- CHART ----------------
st.subheader("📊 Attack Distribution")
st.bar_chart(df["Predicted_Attack"].value_counts())

# ---------------- TABLE ----------------
st.subheader("📋 Sample Predictions")
st.dataframe(df[["Predicted_Attack"]].head(50), use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    🔐 **AI Cyber Threat Prediction & Autonomous Response System**  
    *Academic demonstration project*
    """
)

