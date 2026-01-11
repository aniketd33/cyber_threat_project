import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Cyber Threat Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based Cyber Threat Prediction Dashboard")
st.markdown("Autonomous Cyber Defense (Simulation)")

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload UNSW-NB15 Testing Dataset (.parquet)",
    type=["parquet"]
)

if uploaded_file is None:
    st.warning("Please upload the UNSW_NB15_testing.parquet file")
    st.stop()

df = pd.read_parquet(uploaded_file)
df = df.sample(3000, random_state=42)

# ---------------- TRAIN MODEL (FIRST) ----------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["attack_cat"])
    y = df["attack_cat"]

    X = X.replace("-", np.nan)
    X = X.dropna(axis=1, how="all")   # remove empty columns

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=50,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model

# 🔥 IMPORTANT: model defined here
model = train_model(df)

# ---------------- PREDICTION ----------------
X_pred = df.drop(columns=["attack_cat"])
X_pred = X_pred.replace("-", np.nan)
X_pred = X_pred.dropna(axis=1, how="all")

df["Predicted_Attack"] = model.predict(X_pred)

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

st.markdown("---")
st.markdown(
    """
    🔐 **AI Cyber Threat Prediction & Autonomous Response System**  
    Model is trained dynamically inside the application.  
    Academic Project
    """
)
