import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Cyber Threat Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based Cyber Threat Prediction Dashboard")
st.markdown("Advanced Visual Analytics & Autonomous Response (Stable Version)")

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

# ---------------- NUMERIC FEATURES ONLY ----------------
X = df.select_dtypes(include=[np.number]).copy()
y = df["attack_cat"].astype(str)

X = X.fillna(0)

if X.shape[1] == 0:
    st.error("No numeric features found.")
    st.stop()

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# ---------------- PREDICTION ----------------
df["Predicted_Attack"] = model.predict(X)

# ---------------- THREAT SEVERITY SCORE ----------------
# simple risk score: attack = 1, normal = 0
df["Threat_Score"] = df["Predicted_Attack"].apply(
    lambda x: 0 if x == "Normal" else 1
)

avg_threat_score = df["Threat_Score"].mean()

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Detected Attacks", (df["Predicted_Attack"] != "Normal").sum())
col3.metric("Normal Traffic", (df["Predicted_Attack"] == "Normal").sum())
col4.metric("Avg Threat Score", round(avg_threat_score, 2))

# ---------------- CHART 1: PIE CHART ----------------
st.subheader("🟠 Attack vs Normal Traffic")

fig, ax = plt.subplots(figsize=(3.5,3.5))   # 🔥 size control
df["Predicted_Attack"].value_counts().plot.pie(
    autopct="%1.1f%%",
    startangle=90,
    ylabel="",
    ax=ax
)

ax.axis("equal")   # perfect circle
st.pyplot(fig)


# ---------------- CHART 2: TOP ATTACK TYPES ----------------
st.subheader("🔴 Top 5 Attack Types")

top_attacks = df["Predicted_Attack"].value_counts().head(5)
st.bar_chart(top_attacks)

# ---------------- CHART 3: TIME-BASED CHART ----------------
st.subheader("⏱️ Time-Based Threat Trend (Simulated)")

df["time_index"] = range(len(df))
st.line_chart(
    df.set_index("time_index")["Threat_Score"]
)

# ---------------- SAMPLE TABLE ----------------
st.subheader("📋 Actual vs Predicted (Sample)")
st.dataframe(
    df[["attack_cat", "Predicted_Attack", "Threat_Score"]].head(50),
    use_container_width=True
)

# ---------------- DOWNLOAD REPORT ----------------
st.subheader("⬇️ Download Prediction Report")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV Report",
    data=csv,
    file_name="cyber_threat_report.csv",
    mime="text/csv"
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    🔐 **AI-Based Cyber Threat Prediction & Autonomous Response System**  

    ✔ Machine Learning based attack detection  
    ✔ Threat severity scoring  
    ✔ Advanced visual analytics  
    ✔ Downloadable security report  

    *Academic Project – Stable Deployment Version*
    """
)


