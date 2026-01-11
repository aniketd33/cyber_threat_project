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
    "Upload UNSW-NB15 Dataset (.parquet)",
    type=["parquet"]
)

if uploaded_file is None:
    st.warning("Please upload the UNSW-NB15 parquet file.")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_parquet(uploaded_file)

# small sample for stability
if len(df) > 4000:
    df = df.sample(4000, random_state=42)

if "attack_cat" not in df.columns:
    st.error("Required column 'attack_cat' not found.")
    st.stop()

# ---------------- PREP DATA ----------------
X = df.drop(columns=["attack_cat"]).copy()
y = df["attack_cat"].astype(str)

# clean invalid values
X = X.replace("-", np.nan)

# remove useless columns
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

# ---------------- BUILD PIPELINE ----------------
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ],
    remainder="drop"
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=25,
        random_state=42,
        n_jobs=-1
    ))
])

# ---------------- TRAIN MODEL ----------------
model.fit(X, y)

# ---------------- PREDICT (NO EXTRA LOGIC) ----------------
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
    🔐 **AI-Based Cyber Threat Prediction & Autonomous Response System**  
    - Dataset uploaded dynamically  
    - Model trained and predicted on same cleaned data  
    - Stable preprocessing pipeline  

    *Academic Project*
    """
)
