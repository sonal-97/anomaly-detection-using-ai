import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import shap
import matplotlib.pyplot as plt
import numpy as np

# Streamlit Title
st.title("Cybersecurity Anomaly Detection (Unsupervised - Ignoring Target)")
st.markdown("""
In this version, we *ignore the target column completely*, and use only feature-based anomaly detection using Isolation Forest.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload Access-Log-Anomaly-Detection-Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = [
        "anomaly_score",
        "login_attempt_count",
        "session_duration",
        "user_activity_frequency",
        "target"
    ]

    if all(col in df.columns for col in required_cols):
        all_features = [
            "anomaly_score", 
            "login_attempt_count", 
            "session_duration", 
            "user_activity_frequency"
        ]
        X_all = df[all_features]

        # Feature selection: Select top 2 features using variance or mutual info (without target)
        # Here we use variance just to illustrate purely unsupervised filtering
        variances = X_all.var().sort_values(ascending=False)
        top2_features = variances.index[:2].tolist()
        st.markdown(f"### ✅ Selected Top 2 Features (Highest Variance): *{', '.join(top2_features)}*")

        X = X_all[top2_features]

        # Train Isolation Forest
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)

        # Predict
        df["anomaly_score_model"] = model.decision_function(X)
        df["is_anomaly"] = model.predict(X)
        df["is_anomaly"] = df["is_anomaly"].map({1: 0, -1: 1})  # 1: normal, -1: anomaly

        # Show data
        st.subheader("Preview: Data with Anomaly Predictions")
        st.dataframe(df[top2_features + ["is_anomaly", "anomaly_score_model"]])

        # Scatter plot (points only)
        st.subheader("Scatter Plot of Selected Features (Purely Model Predictions)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        normal = df[df["is_anomaly"] == 0]
        anomaly = df[df["is_anomaly"] == 1]

        ax1.scatter(normal[top2_features[0]], normal[top2_features[1]],
                    c='green', label='Predicted Normal', alpha=0.6, edgecolors='k')
        ax1.scatter(anomaly[top2_features[0]], anomaly[top2_features[1]],
                    c='red', label='Predicted Anomaly', alpha=0.8, edgecolors='k', marker='x')
        ax1.set_xlabel(top2_features[0])
        ax1.set_ylabel(top2_features[1])
        ax1.set_title("Scatter Plot (Ignoring Target Labels)")
        ax1.legend()
        st.pyplot(fig1)

        # Isolation Forest decision boundary plot
        st.subheader("Isolation Forest Decision Boundary Plot")
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        # Tight mesh grid
        x_min, x_max = X[top2_features[0]].min() - 0.05, X[top2_features[0]].max() + 0.05
        y_min, y_max = X[top2_features[1]].min() - 0.5, X[top2_features[1]].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        mesh_points = pd.DataFrame({
            top2_features[0]: xx.ravel(),
            top2_features[1]: yy.ravel(),
        })
        Z = model.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)

        ax2.contour(xx, yy, Z, levels=[0], linewidths=2, colors='blue', linestyles="--")
        ax2.set_xlabel(top2_features[0])
        ax2.set_ylabel(top2_features[1])
        ax2.set_title("Isolation Forest Decision Boundary (Unsupervised)")
        st.pyplot(fig2)

        # SHAP
        st.subheader("SHAP Feature Importance (Unsupervised)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        fig3 = plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig3)

        # Sorted anomalies
        st.subheader("Sorted Anomalies (Most Severe on Top)")
        sorted_anomalies = df[df["is_anomaly"] == 1].copy()
        sorted_anomalies = sorted_anomalies.sort_values(by="anomaly_score_model")
        st.dataframe(sorted_anomalies[top2_features + ["anomaly_score_model"]])

    else:
        st.error("The uploaded CSV is missing one or more required columns:\n" + ", ".join(required_cols))
else:
    st.info("Please upload a CSV file to begin.")
