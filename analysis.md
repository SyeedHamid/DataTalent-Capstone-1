# Analysis and Interpretation

## Insights from Visualizations

### Long-Term Trends
- Median temperature anomalies from 1880 to 2023 show a clear upward trajectory, especially post-1970.
- Rolling 5-year averages smooth short-term noise and highlight persistent warming phases.

### Anomaly Detection
- S-H-ESD flagged outliers in residuals after seasonal decomposition.
- These anomalies often correspond to periods of rapid climate shifts or unusual seasonal behavior.

### Model Performance
- Linear Regression captured the general trend but underperformed on short-term fluctuations.
- Decision Tree Regressor adapted better to variability but showed signs of overfitting.
- Residual histograms confirmed model bias and variance trade-offs.

### Geospatial Data
- NetCDF ensemble mean data was loaded using `xarray` and prepared for spatial analysis.
- While full mapping was not implemented, the structure supports future regional anomaly detection.

## Assumptions

- **Stationarity**: Assumed after ADF testing and smoothing. Non-stationary segments were decomposed.
- **Data Integrity**: Relied on Kaggle and HadCRUT5 datasets as accurate and representative.
- **Model Simplicity**: Chose interpretable models over complex ones to prioritize clarity and reproducibility.

## Conclusions

- Global temperatures have risen significantly, with anomalies becoming more frequent and extreme.
- Statistical methods like S-H-ESD are effective for identifying deviations in seasonal climate data.
- Simple ML models offer useful approximations but are limited in capturing nonlinear climate dynamics.
- Geospatial data integration opens pathways for more granular climate analysis in future work.
