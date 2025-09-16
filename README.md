**
DataTalent-Capstone-1
Global Temperature Anomaly Detection
Capstone Project – DataTalent Program**

**Overview**
This project analyzes global surface temperature anomalies to detect significant climate shifts using time series analysis, statistical anomaly detection, and supervised machine learning. The goal is to identify long-term warming trends and abrupt month-over-month changes, and to validate these findings using ensemble models.
Dataset

- Source: Kaggle (HadCRUT4/5 ensemble dataset)- https://www.kaggle.com/datasets/rupindersinghrana/global-temperature-anomalies
- Coverage: Monthly global surface temperature anomalies from 1880 to 2023
- Format: CSV (cleaned and transformed from ensemble realizations)
- Features: Year, Month, Median anomaly, Rolling average, Month-over-month change

  
**Tools and Technologies**
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Azure Blob Storage
- Jupyter Notebook
- Azure ML Studio (optional for deployment)

  
**Methodology**
- Exploratory Data Analysis (EDA)
- Feature engineering: rolling averages, anomaly deltas
- Statistical detection of top anomaly shifts
- Supervised regression using:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Model evaluation using R² and MSE
- Feature importance analysis for interpretability

  
**Key Results**
- Detected top 10 abrupt temperature shifts across the dataset
- Ensemble models (Gradient Boosting, Random Forest) achieved R² > 0.88
- Year was the dominant feature, confirming long-term warming trend
- Visualizations revealed seasonal variability and distribution of anomalies
- Pipeline is reproducible, scalable, and ready for forecasting extensions

Let me know if you want this formatted for your GitHub README.md, or turned into a slide for your final presentation. You're presenting a clean, rigorous analysis—exactly what a capstone should be.
