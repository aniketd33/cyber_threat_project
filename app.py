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
    st.warning("Please upload the UNSW_NB15 parquet file to continue.")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_parquet(uploaded_file)

# take small sample to avoid memory / timeout issues
if len(df) > 5000:
    df = df.sample(5000, random_state=42)

# check target column
if "attack_cat" not in df.columns:
    st.error("Column 'attack_cat' not found in dataset.")
    st.stop()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["attack_cat"])
    y = df["attack_cat"].astype(str)

    # replace invalid values
    X = X.replace("-", np.nan)

    # 🔥 CRITICAL FIXES (NO MORE ERRORS)
    # 1. remove fully empty columns
    X = X.dropna(axis=1, how="all")

    # 2. remove constant-value columns
    X = X.loc[:, X.nunique(dropna=True) > 1]

    # identify column types
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    # preprocessing
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
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
            n_estimators=30,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model, X.columns  # return feature columns also

model, feature_cols = train_model(df)

# ---------------- PREDICTION ----------------
X_pred = df.drop(columns=["attack_cat"])
X_pred = X_pred.replace("-", np.nan)

# keep only trained features
X_pred = X_pred[feature_cols]

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

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    🔐 **AI-Based Cyber Threat Prediction & Autonomous Response System**  
    - Dataset uploaded dynamically  
    - Model trained inside the app  
    - Robust preprocessing to avoid runtime errors  

    *Academic Project*
    """
)
