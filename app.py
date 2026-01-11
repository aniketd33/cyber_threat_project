import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Cyber Threat Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based Cyber Threat Prediction Dashboard")
st.markdown("Stable Version (Numeric Features Only)")

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload UNSW-NB15 Dataset (.parquet)",
    type=["parquet"]
)

if uploaded_file is None:
    st.warning("Please upload the UNSW-NB15 parquet file.")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_parquet(uploaded_file)

# sample to avoid crash
if len(df) > 5000:
    df = df.sample(5000, random_state=42)

if "attack_cat" not in df.columns:
    st.error("Column 'attack_cat' not found in dataset.")
    st.stop()

# ---------------- SIMPLE CLEANING ----------------
# keep ONLY numeric columns
X = df.select_dtypes(include=[np.number]).copy()
y = df["attack_cat"].astype(str)

# fill NaN with 0 (SAFE)
X = X.fillna(0)

# safety check
if X.shape[1] == 0:
    st.error("No numeric features available in dataset.")
    st.stop()

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ---------------- PREDICT ----------------
df["Predicted_Attack"] = model.predict(X)

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
st.dataframe(
    df[["attack_cat", "Predicted_Attack"]].head(50),
    use_container_width=True
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    🔐 **AI Cyber Threat Prediction & Autonomous Response System**  
    - Numeric features only  
    - No complex preprocessing  
    - Stable deployment version  

    *Academic Project*
    """
)
