# 🌍 AirSense-AI

A Machine Learning-based Air Quality Prediction and Historical AQI Analysis System built using Streamlit. The application predicts the Air Quality Index (AQI) from pollutant concentrations and provides interactive visualizations to compare historical AQI and pollutant trends across four major Indian metropolitan cities: Delhi, Mumbai, Chennai, and Kolkata.

---

# 🚀 Features

- Historical AQI Analysis for Four Cities
- AQI Prediction using Machine Learning
- City-wise AQI Comparison Dashboard
- Pollutant Comparison Charts
- Monthly AQI Trend Analysis
- Monthly Pollutant Trend Analysis
- Health Advisory Based on Predicted AQI
- Interactive Streamlit Dashboard
- Plotly Visualizations

---

# 🧠 Key Concepts Used

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Supervised Machine Learning
- Regression Analysis
- Random Forest Regression
- XGBoost Regression
- Model Evaluation and Comparison
- Data Visualization
- Streamlit Web Application Development

---

# 🛠️ Tools & Technologies

**Language:** Python 3.14

**Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Joblib, Plotly

**Frontend:** Streamlit, Streamlit Option Menu

**Development Environment:** Visual Studio Code

**Dataset:** Historical 2025 Air Quality Dataset (Delhi, Mumbai, Chennai, Kolkata)

---

# 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|------|----:|------:|---------:|
| Linear Regression | 22.42 | 29.78 | 0.9088 |
| Random Forest | **2.51** | **7.91** | **0.9936** |
| XGBoost | 3.34 | 9.70 | 0.9903 |

**Final Model:** Random Forest Regressor

---

# 📂 Project Structure

```text
AirSense-AI/
│
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
│
├── data/
│   └── cleaned_air_quality.csv  # Historical AQI dataset
│
├── models/
│   └── aqi_model.pkl            # Trained Random Forest model
│
├── src/
│   ├── app.py                   # Streamlit application
│   ├── preprocess.py            # Data preprocessing
│   └── train_model.py           # Model training and evaluation
│
├── requirements.txt             # Project dependencies
├── .gitignore                   # Git ignored files
└── README.md                    # Project documentation
```

---

# ▶️ How to Run Locally

1. Clone this repository.

```bash
git clone https://github.com/<your-username>/AirSense-AI.git
```

2. Navigate to the project directory.

```bash
cd AirSense-AI
```

3. Create a virtual environment.

```bash
python -m venv venv
```

4. Activate the virtual environment.

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

5. Install the required libraries.

```bash
pip install -r requirements.txt
```

6. Train the machine learning model.

```bash
python src/train_model.py
```

7. Launch the Streamlit application.

```bash
streamlit run src/app.py
```

The application will automatically open in your default web browser.

---
