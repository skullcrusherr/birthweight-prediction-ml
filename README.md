
# 🍼 Birthweight Prediction System  
### Multi‑Model Machine Learning Healthcare Project  

---

## 📌 Project Overview

This project predicts **neonatal birthweight (in kilograms)** using maternal and parental health attributes.  
It implements and compares multiple machine learning models to evaluate performance on structured healthcare data.

The system includes:

- Multiple regression models
- Model comparison dashboard
- Login & user authentication
- Prediction history tracking
- Optional retraining system
- Interactive Streamlit web interface

---

## 🧠 Machine Learning Models Implemented

The following models were trained and evaluated:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- XGBoost Regressor  
- Semantic KNN (Cosine Similarity)

Performance metrics used:

- MSE (Mean Squared Error)  
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

---

## 📊 Features Used for Prediction

The model uses structured clinical attributes including:

- Gestation Period  
- Maternal Age (mage)  
- Maternal Pre‑Pregnancy Weight (mppwt)  
- Smoking Status  
- Father’s Age (fage)  
- Father’s Education Years (fedyrs)  
- Cigarette Consumption (fnocig)  
- Parental Height  
- Maternal Age ≥ 35 Indicator (mage35)  
- Head Circumference  
- Baby Length  

Target Variable: **Birthweight (continuous regression output)**

---

## 🖥 Application Features

✔ Secure Login & Registration  
✔ Model Selection  
✔ Real‑time Birthweight Prediction  
✔ Model Performance Evaluation  
✔ Prediction History Tracking (SQLite)  
✔ Optional Model Retraining  
✔ Clean Visualization Dashboard  

---

## 🏗 Project Structure

```
.
├── app.py
├── train_model.py
├── birth_weight.csv
├── requirements.txt
├── model/
│   ├── lr_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── dt_model.pkl
│   ├── semantic_knn_model.pkl
│   ├── imputer.pkl
│   └── scaler.pkl
├── app.db
├── users.db
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Python 3.11

Ensure Python 3.11 is installed.

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

### 3️⃣ Activate Virtual Environment

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📈 Model Insights

The ensemble models (Random Forest and XGBoost) showed similar performance to Linear Regression, indicating limited nonlinear signal in the dataset. The Semantic KNN model performed comparatively weaker due to the structured numeric nature of the data.

Negative R² values suggest the model performance is close to a baseline mean predictor, highlighting potential scope for:

- Feature engineering  
- Hyperparameter tuning  
- Additional clinical features  
- Cross‑validation  

---

## 🔐 Database

SQLite databases used:

- `users.db` → Authentication storage  
- `app.db` → Prediction history  

---

## 🎓 Academic Value

This project demonstrates:

- End‑to‑end ML pipeline implementation  
- Model comparison framework  
- Deployment via Streamlit  
- Integration of database systems  
- Reproducible saved model workflow  

---

## 🧑‍💻 Author

Namith N

---

⭐ If you find this project useful, consider starring the repository!
