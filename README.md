# 🛡️ AI-Based Cyber Threat Prediction & Autonomous Response System

## 📌 Project Overview
This project implements an AI-powered Cyber Threat Detection System using Machine Learning.
It detects different cyber attacks, identifies anomalous network traffic, and performs a
simulated autonomous response by blocking malicious IP addresses.
A Streamlit dashboard is provided for visualization and demonstration.

---

## 🎯 Aim
To build an intelligent cyber security system that can automatically detect and respond to
network-based cyber attacks using AI techniques.

---

## 🧠 Objectives
- Load and analyze a real-world intrusion detection dataset (UNSW-NB15)
- Preprocess network traffic data safely
- Train a Machine Learning model for attack classification
- Detect anomalous traffic using anomaly detection
- Implement autonomous response (IP blocking simulation)
- Visualize results using a dashboard

---

## 🗂️ Project Structure
cyber_threat_project/
│
├── app.py                     # Streamlit dashboard
├── README.md                  # Project documentation
├── unsw_rf_model.pkl          # Trained ML pipeline
├── label_encoder.pkl          # Label encoder for attacks
│
└── dataset/
    ├── UNSW_NB15_training.parquet
    └── UNSW_NB15_testing.parquet
---

## 📊 Dataset Used
**UNSW-NB15 Dataset**

- Realistic network traffic dataset
- Contains normal and attack traffic
- Multiple attack categories

**Target column:** `attack_cat`

---

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---
## 🚨 Threat Detection

An **Isolation Forest** model is used to detect anomalous (suspicious) network traffic.
This model works without using attack labels and identifies traffic patterns that
deviate from normal behavior.

Each network record is classified as:
- **Normal** – Regular and expected traffic
- **Threat** – Abnormal or suspicious traffic pattern

---

## 🤖 Autonomous Response (Simulation)

The system implements an autonomous response mechanism based on detected threats.

### Response Logic:
- Known attack types are **automatically blocked**
- Traffic detected as anomalous is **blocked**
- Normal traffic is **allowed**

⚠️ **Note:**  
Blocking is **simulated only** for academic purposes.  
No real firewall rules or operating system configurations are modified.

---
## 📊 Dashboard

A **Streamlit-based interactive dashboard** is developed to visualize the outputs of the
AI-based cyber threat detection system.

### Dashboard Features:
- Displays total network traffic records
- Shows number of detected attacks
- Visualizes attack distribution using charts
- Presents prediction results in tabular format

### Run Dashboard:
```bash
streamlit run app.py
http://localhost:8501
