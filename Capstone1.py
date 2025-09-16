# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Azure Blob Import ---
def load_data_from_blob(connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    stream = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(stream))
    logging.info("Data loaded from Azure Blob Storage.")
    return df

# --- Preprocessing ---
def preprocess(df):
    ensemble_cols = [col for col in df.columns if col.startswith("Realization_")]
    df['Date'] = pd.to_datetime(df['t'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Median'] = df[ensemble_cols].median(axis=1)
    df['Rolling_Avg'] = df['Median'].rolling(window=12).mean()
    df['MoM_Change'] = df['Median'].diff()
    df.dropna(subset=['Median'], inplace=True)
    logging.info("Preprocessing complete.")
    return df

# --- Visualizations ---
def plot_visualizations(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Median'], color='darkblue', linewidth=2)
    plt.title("Global Temperature Anomalies Over Time")
    plt.xlabel("Year")
    plt.ylabel("Anomaly (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Median'], label='Monthly')
    plt.plot(df['Date'], df['Rolling_Avg'], color='red', label='12-Month Avg', linewidth=2)
    plt.title("Smoothed Temperature Anomalies")
    plt.xlabel("Year")
    plt.ylabel("Anomaly (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.histplot(df['Median'], bins=30, kde=True, color='teal')
    plt.title("Distribution of Monthly Temperature Anomalies")
    plt.xlabel("Median Anomaly (°C)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month', y='Median', data=df, palette='coolwarm')
    plt.title("Seasonal Distribution of Temperature Anomalies")
    plt.xlabel("Month")
    plt.ylabel("Median Anomaly (°C)")
    plt.tight_layout()
    plt.show()

# --- Anomaly Detection ---
def plot_top_mom_changes(df):
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

# --- Modeling ---
def train_models(df):
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
        logging.info(f"{name} - R2: {r2:.4f}, MSE: {mse:.4f}")
        joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")

    return models, results

# --- Feature Importance ---
def plot_feature_importance(models, features):
    for name in ["Random Forest", "Gradient Boosting"]:
        model = models[name]
        importances = model.feature_importances_
        plt.figure(figsize=(6, 4))
        plt.bar(features, importances, color='skyblue')
        plt.title(f"{name} Feature Importances")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- Main Execution ---
def main():
    # Replace with your Azure credentials
    connection_string = "your_connection_string"
    container_name = "your_container"
    blob_name = "your_dataset.csv"

    df_raw = load_data_from_blob(connection_string, container_name, blob_name)
    df = preprocess(df_raw)
    plot_visualizations(df)
    plot_top_mom_changes(df)
    models, results = train_models(df)
    plot_feature_importance(models, ['Year', 'Month'])

if __name__ == "__main__":
    main()
