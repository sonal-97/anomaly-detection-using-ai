import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate sample user activity data
np.random.seed(42)
data = {
    "login_attempts": np.random.poisson(3, 100),
    "session_duration": np.random.normal(300, 50, 100),
    "bytes_sent": np.random.normal(1000, 200, 100),
    "failed_logins": np.random.binomial(5, 0.1, 100)
}
df = pd.DataFrame(data)

# Add some anomalies manually
df.iloc[5] = [15, 900, 4000, 5]    # suspicious user
df.iloc[25] = [12, 700, 3000, 4]   # another one

# Step 2: Train Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df)

# Step 3: Predict anomalies (use only original columns)
features = ["login_attempts", "session_duration", "bytes_sent", "failed_logins"]
df["anomaly_score"] = model.decision_function(df[features])
df["is_anomaly"] = model.predict(df[features])
df["is_anomaly"] = df["is_anomaly"].map({1: 0, -1: 1})  # 1 for anomaly

# Step 4: Use SHAP for Explainable AI
explainer = shap.Explainer(model, df[features])
shap_values = explainer(df[features])

# Plot summary of SHAP values
shap.summary_plot(shap_values, df[features])

# Optional: Show one specific anomaly’s explanation
anomalous_index = df[df["is_anomaly"] == 1].index[0]
shap.plots.waterfall(shap_values[anomalous_index])
