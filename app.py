import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Page title
st.title("💡 AI-Based Data Quality Intelligence System")
st.write("Upload any dataset to automatically evaluate its quality using AI + ML.")

# Sidebar
st.sidebar.title("📌 Navigation")
st.sidebar.info("Upload a CSV or Excel file to begin the analysis.")

# File uploader (CSV + Excel)
uploaded_file = st.file_uploader("Upload Dataset File", type=["csv", "xlsx"])

if uploaded_file is not None:

    # Read CSV or Excel
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.markdown("### 🔍 Dataset Preview")
    st.write(data.head())

    # Remove unnecessary index column if exists
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    # Missing values
    missing = data.isnull().sum().sum()

    # Duplicate rows
    duplicates = data.duplicated().sum()
    data = data.drop_duplicates()

    # Select numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Fill missing values for ML
    numeric_data = numeric_data.fillna(numeric_data.mean())

    # Handle case: no numeric columns
    if numeric_data.shape[1] == 0:
        st.error("❌ No numeric columns found in the dataset. Cannot perform anomaly detection.")
    else:
        # Isolation Forest Model
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(numeric_data)
        data['anomaly'] = model.predict(numeric_data)

        anomalies = (data['anomaly'] == -1).sum()

        # Quality Score
        total_records = len(data)
        error_count = missing + anomalies
        quality_score = ((total_records - error_count) / total_records) * 100

        st.markdown("### 📊 Data Quality Report")
        st.write("🔸 Missing Values:", missing)
        st.write("🔸 Duplicate Rows:", duplicates)
        st.write("🔸 Anomalies Detected:", anomalies)

        # Quality score with color status
        st.markdown("### 📈 Quality Score Analysis")
        if quality_score >= 90:
            st.success(f"✅ Excellent Data Quality: {round(quality_score, 2)}%")
        elif quality_score >= 75:
            st.warning(f"⚠️ Moderate Data Quality: {round(quality_score, 2)}%")
        else:
            st.error(f"❌ Poor Data Quality: {round(quality_score, 2)}%")

        # Visualization Section
        st.markdown("### 📉 Anomaly Detection Visualization")
        normal_count = (data['anomaly'] == 1).sum()
        anomaly_count = (data['anomaly'] == -1).sum()

        fig = plt.figure()
        plt.bar(["Normal", "Anomaly"], [normal_count, anomaly_count])
        plt.ylabel("Count")
        plt.title("Anomaly Detection Result")
        st.pyplot(fig)

        # Downloadable Report
        st.markdown("### 📥 Download Analysis Report")
        report = f"""
AI-Based Data Quality Report

Missing Values: {missing}
Duplicate Rows: {duplicates}
Anomalies Detected: {anomalies}
Quality Score: {round(quality_score,2)}%

Thank you for using the Data Quality Intelligence System!
        """

        st.download_button(
            label="⬇ Download Report",
            data=report,
            file_name="data_quality_report.txt",
            mime="text/plain"
        )