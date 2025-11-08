## Power Demand Forecasting using Machine Learning & Deep Learning  

> Forecasting hourly **PJM Interconnection (PJM)** power demand for the next 30 days using hybrid Machine Learning and Deep Learning models — deployed via Streamlit for interactive forecasting and analysis.  

---

### Overview  
The project focuses on **predicting hourly electricity consumption (MW)** for the PJM regional transmission network, which serves multiple U.S. states.  
The models were trained using **16 years of hourly data (2002–2018)** to help improve energy load management, grid reliability, and future demand planning.

---

### Models Implemented  
| Category | Models | Description |
|-----------|---------|-------------|
| 📈 Statistical | SARIMA | Captures temporal patterns using autoregression |
| 🧭 Additive | Prophet / NeuralProphet | Handles trend, seasonality & holiday effects |
| 🧩 Deep Learning | LSTM / TCN | Learns temporal dependencies via sequence modeling |
| 🌲 Ensemble ML | XGBoost / LightGBM | Gradient boosting for tabular time series data |

---

<details>
<summary>📊 <b>Data Description</b></summary>

- **Source:** PJM Hourly Energy Consumption Dataset  
- **Duration:** 2002 – 2018  
- **Frequency:** Hourly  
- **Target Variable:** `PJMW_MW`  
- **Total Records:** 143,232  
- **Statistical Summary:**
  - Mean = 5602 MW  
  - Std = 979 MW  
  - Min = 487 MW  
  - Max = 9594 MW  
  - Missing Values = 0  

</details>

---

### Exploratory Insights  
- Power demand exhibits **strong daily, weekly, and annual seasonality**.  
- **Winter and Summer peaks** due to heating and cooling usage.  
- **Weekends and holidays** show reduced load due to lower industrial activity.  
- Consumption patterns show **cyclic stability with periodic surges**.  

---

### Deployment  
The final deployment uses a **Streamlit web application** built with the **TCN model** for real-time hourly power demand forecasting.  

**App Features:**  
✅ Adjustable forecast horizon (1 – 30 days)  
✅ Visualization of last 4 years of historical demand  
✅ Forecasted hourly series and daily summary  
✅ Downloadable forecast results (CSV format)  

---

### Forecast Results  
| Model | MAE (MW) | RMSE (MW) | R² | Remarks |
|--------|-----------|-----------|----|----------|
| SARIMA | 798.6 | 1020.3 | -0.05 | Weak baseline |
| Prophet | 922.3 | 1332008.1 | -0.34 | Overfit on trend |
| NeuralProphet | 567.1 | 527822.5 | 0.46 | Moderate accuracy |
| LSTM | 373.0 | 464.2 | 0.78 | Strong temporal learner |
| XGBoost | 67.2 | 7828.9 | 0.99 | Excellent ensemble |
| LightGBM | 65.2 | 7250.9 | 0.99 | Best ensemble model |
| **TCN** | **81.8** | **10752.0** | **0.98** | ✅ Best deep model |

---

### Key Insights from Deployment  
- The **TCN model** successfully forecasts **30-day hourly load** with minimal deviation.  
- The **forecasted pattern aligns with seasonal cyclic behavior** seen in the historical data.  
- **Daily summary tables** (mean + std dev) help identify stability and volatility in power demand.  
- This makes it suitable for **energy management dashboards** and **load balancing decisions**.  

---

### Tech Stack  
**Languages:** Python  
**Libraries:** pandas, numpy, scikit-learn, keras, tcn, prophet, lightgbm, xgboost, matplotlib, plotly, streamlit  
**Deployment:** Streamlit Web Application  

---

### Challenges Faced  
1. Handling large hourly datasets and ensuring temporal continuity  
2. Multi-seasonal pattern detection and high-resolution time windowing  
3. Generating stable **multi-step (720 hours)** forecasts  
4. Balancing **model interpretability vs. accuracy**  
5. Integrating deep learning models into **real-time web deployment**

---

### Project Description  
This project builds a complete end-to-end pipeline for **hourly energy demand forecasting** of the PJM Interconnection grid.  
It integrates **data preprocessing, exploratory analysis, model comparison, and TCN-based deployment**, achieving over **98–99% forecast accuracy** on unseen data.  
The final product is an interactive **Streamlit dashboard** for 30-day hourly predictions with exportable results and analytical summaries.

---

### Preview  
![Streamlit Dashboard](screencapture-localhost-8501-2025-07-13-16_19_25%20top.PNG)  
![Forecast Table](screencapture-localhost-8501-2025-07-13-16_19_25%20bot.PNG)

---
<h3 align="center">By</h3> 

<h4 align="center">Sucharita Lakkavajhala</h4>
<h4 align="center">Shyam Kumar Kampelly</h4>
<h4 align="center">Uday Kumar Barigela</h4>
<h4 align="center">Pravalika Challuri</h4>

---

⭐ *If you found this project helpful, give it a star and share your feedback!*
