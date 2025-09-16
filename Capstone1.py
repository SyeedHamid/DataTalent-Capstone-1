# -----------------------------------------------
# 1. Imports
# -----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from azure. storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
# -----------------------------------------------
# 2. Load Dataset from Azure Blob Storage
# -----------------------------------------------
connection_string = "your_connection_string"
container_name = "your_container"
blob_name = "your_dataset.csv"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

stream = io.BytesIO()
blob_client.download_blob().readinto(stream)
stream.seek(0)
df = pd.read_csv(stream)

# -----------------------------------------------
# 3. Data Preprocessing
# -----------------------------------------------

ensemble_cols = [col for col in df.columns if col.startswith("Realization_")]

df['Date'] = pd.to_datetime(df['t'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df['Median'] = df[ensemble_cols].median(axis=1)
df['Rolling_Avg'] = df['Median'].rolling(window=12).mean()
df['MoM_Change'] = df['Median'].diff()

# -----------------------------------------------
# 4. Visualizations
# -----------------------------------------------

# Global temperature anomalies over time
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Median'], color='darkblue', linewidth=2)
plt.title("Global Temperature Anomalies Over Time")
plt.xlabel("Year")
plt.ylabel("Anomaly (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Smoothed anomalies using rolling average
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Median'], label='Monthly')
plt.plot(df['Date'], df['Rolling_Avg'], color='red', label='12-Month Avg', linewidth=2)
plt.title("Smoothed Temperature Anomalies")
plt.xlabel("Year")
plt.ylabel("Anomaly (°C)")
plt.legend()
plt.tight_layout()
plt.show()

# Distribution of monthly anomalies
plt.figure(figsize=(12, 5))
sns.histplot(df['Median'], bins=30, kde=True, color='teal')
plt.title("Distribution of Monthly Temperature Anomalies")
plt.xlabel("Median Anomaly (°C)")
plt.tight_layout()
plt.show()

# Monthly boxplot of anomalies
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Median', data=df, palette='coolwarm')
plt.title("Monthly Distribution of Temperature Anomalies")
plt.xlabel("Month")
plt.ylabel("Median Anomaly (°C)")
plt.tight_layout()
plt.show()


# -----------------------------------------------
# 5. Top 10 Month-over-Month Change Events
# -----------------------------------------------


top_changes = df[['Date', 'MoM_Change']].dropna().copy()
top_changes['Abs_Change'] = top_changes['MoM_Change'].abs()
top10 = top_changes.sort_values(by='Abs_Change', ascending=False).head(10)

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['MoM_Change'], color='gray', linewidth=1)
for _, row in top10.iterrows():
    color = 'green' if row['MoM_Change'] > 0 else 'red'
    plt.axvline(row['Date'], color=color, linestyle='--', linewidth=1)
    plt.scatter(row['Date'], row['MoM_Change'], color=color, s=50)
plt.title("Top 10 Month-over-Month Temperature Change Events")
plt.xlabel("Date")
plt.ylabel("Change in Anomaly (°C)")
plt.tight_layout()
plt.show()


# -----------------------------------------------
# 6. Model Training and Evaluation
# -----------------------------------------------

x = df[['Year', 'Month']]
y = df['Median']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = {"R2": r2, "MSE": mse}
    print(f"{name} - R2: {r2:.4f}, MSE: {mse:.4f}")


# -----------------------------------------------
# 7. Model Performance Summary
# -----------------------------------------------

sorted_results = sorted(results.items(), key=lambda x: x[1]["R2"], reverse=True)

print("\nModel Performance Comparison:")
for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{i}. {name} - R2: {metrics['R2']:.4f}, MSE: {metrics['MSE']:.4f}")



# -----------------------------------------------
# 8. Feature Importance (Tree Models Only)
# -----------------------------------------------

for name in ["Random Forest", "Gradient Boosting"]:
    model = models[name]
    importances = model.feature_importances_
    plt.figure(figsize=(6, 4))
    plt.bar(x.columns, importances, color='skyblue')
    plt.title(f"{name} Feature Importances")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

