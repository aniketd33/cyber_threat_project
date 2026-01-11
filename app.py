import streamlit as st
import pandas as pd
import joblib
import gzip
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Cyber Threat Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CACHE LOADERS ----------------
@st.cache_resource
def load_model():
    # If model is compressed (.pkl.gz)
    try:
        with gzip.open("unsw_rf_model.pkl.gz", "rb") as f:
            return joblib.load(f)
    except FileNotFoundError:
        # fallback if not compressed
        return joblib.load("unsw_rf_model.pkl")

@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")

@st.cache_data
def load_data():
    df = pd.read_parquet("dataset/UNSW_NB15_testing.parquet")
    # 🔥 VERY IMPORTANT: use sample to avoid loading freeze
    return df.sample(5000, random_state=42)

# ---------------- LOAD ASSETS ----------------
model = load_model()
label_encoder = load_label_encoder()
df = load_data()

# ---------------- TITLE ----------------
st.title("🛡️ AI-Based Cyber Threat Prediction Dashboard")
st.markdown("Autonomous Cyber Defense (Simulation)")

# ---------------- DATA PREP ----------------
if "attack_cat" in df.columns:
    X = df.drop(columns=["attack_cat"])
    y_true = df["attack_cat"]
else:
    X = df.copy()
    y_true = None

# Replace invalid symbols
X = X.replace("-", np.nan)

# ---------------- PREDICTION ----------------
pred_encoded = model.predict(X)
pred_labels = label_encoder.inverse_transform(pred_encoded)

df["Predicted_Attack"] = pred_labels

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Detected Attacks", (df["Predicted_Attack"] != "Normal").sum())
col3.metric("Normal Traffic", (df["Predicted_Attack"] == "Normal").sum())

# ---------------- DISTRIBUTION ----------------
st.subheader("📊 Attack Distribution")
attack_counts = df["Predicted_Attack"].value_counts()
st.bar_chart(attack_counts)

# ---------------- RESULTS TABLE ----------------
st.subheader("📋 Prediction Results (Sample)")
st.dataframe(
    df[["Predicted_Attack"]].head(50),
    use_container_width=True
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    🔐 **AI Cyber Threat Prediction & Autonomous Response System**  
    - ML-based attack classification  
    - Anomaly-aware decision making  
    - Simulated autonomous response  
    - Streamlit interactive dashboard  

    *Developed for academic demonstration purposes.*
    """
)
