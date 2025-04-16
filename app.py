import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Step 1: Generate sample user activity data (e.g., login attempts, bytes sent, etc.)
np.random.seed(42)
data = {
    "login_attempts": np.random.poisson(3, 100),
    "session_duration": np.random.normal(300, 50, 100),
    "bytes_sent": np.random.normal(1000, 200, 100),
    "failed_logins": np.random.binomial(5, 0.1, 100)
}

df = pd.DataFrame(data)

# Add some anomalies
df.iloc[5] = [15, 900, 4000, 5]  # suspicious user
df.iloc[25] = [12, 700, 3000, 4]  # another one

# Step 2: Train Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)

# Fit model on original data
X = df.copy()  # Create a copy to preserve the original data
model.fit(X)

# Step 3: Predict anomalies and add anomaly_score and is_anomaly columns
df["anomaly_score"] = model.decision_function(X)
df["is_anomaly"] = model.predict(X)
df["is_anomaly"] = df["is_anomaly"].map({1: 0, -1: 1})  # Convert 1/-1 to 0/1

# Step 4: Use SHAP for Explainable AI
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # Use original features only for SHAP

# Step 5: Visualize SHAP values using Streamlit
st.title('Anomaly Detection with Isolation Forest')
st.write('This is a simple anomaly detection model using Isolation Forest.')

# Display the data with the predictions
st.subheader('User Activity Data with Anomaly Predictions')
st.write(df)

# Filter anomalies for display
anomalies = df[df['is_anomaly'] == 1]
st.subheader(f'Detected Anomalies ({len(anomalies)} records)')
if not anomalies.empty:
    st.write(anomalies)
else:
    st.write("No anomalies detected")

# Create a figure for the SHAP summary plot
st.subheader('SHAP Summary Plot')
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)

# Display anomalies with highest impact
if not anomalies.empty:
    st.subheader('Analysis of Top Anomaly')
    anomaly_index = anomalies.index[0]
    
    # For force plot instead of waterfall (compatible with Isolation Forest)
    st.write("SHAP Force Plot for Top Anomaly:")
    # First, convert shap values to a dense format if they're sparse
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
        st_shap = st.pyplot(shap.force_plot(
            explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
            shap_values[anomaly_index,:],
            X.iloc[anomaly_index,:], 
            feature_names=X.columns,
            matplotlib=True,
            show=False
        ))
    
    # Show the anomaly details
    st.write(f"Anomaly record (index {anomaly_index}):")
    st.write(df.loc[anomaly_index])
    
    # Create a bar chart showing feature values compared to mean
    st.subheader('Feature Comparison for Top Anomaly')
    anomaly_data = df.loc[anomaly_index, ["login_attempts", "session_duration", "bytes_sent", "failed_logins"]]
    mean_data = df[["login_attempts", "session_duration", "bytes_sent", "failed_logins"]].mean()
    
    comparison_df = pd.DataFrame({
        'Anomaly': anomaly_data.values,
        'Dataset Average': mean_data.values
    }, index=anomaly_data.index)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax)
    plt.title('Anomaly vs Average Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Feature importance plot based on SHAP values
st.subheader('Feature Importance')
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig)

# Add a heatmap visualization of the correlation between features
st.subheader('Feature Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[["login_attempts", "session_duration", "bytes_sent", "failed_logins"]].corr(), 
            annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Add an anomaly score distribution
st.subheader('Anomaly Score Distribution')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['anomaly_score'], kde=True, ax=ax)
plt.axvline(x=df[df['is_anomaly']==1]['anomaly_score'].min(), color='red', 
            linestyle='--', label='Anomaly Threshold')
plt.legend()
plt.title('Distribution of Anomaly Scores')
st.pyplot(fig)