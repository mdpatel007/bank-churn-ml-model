# **Bank Churn Prediction Model**

## 📑 Table of Contents
- [Overview](#overview)  
- [Problem Statement](#problem-statement)  
- [Solution](#solution)  
- [Technical Stack](#technical-stack)  
- [Dataset](#dataset)  
- [Key Features](#key-features)  
- [Data Preprocessing](#data-preprocessing)  
- [Model Performance](#model-performance)  
- [Project Structure](#project-structure)  
- [How to Run](#how-to-run)  
- [Future Enhancements](#future-enhancements)  
- [Acknowledgments](#acknowledgments)  
- [Live Demo](#live-demo)  

---

## 🧠 Overview
This project develops and implements a machine learning model to predict customer churn in a banking context. By identifying customers at high risk of churning, banks can proactively implement retention strategies, minimize attrition, and improve customer lifetime value and profitability.

---

## ❓ Problem Statement
Customer churn is a significant challenge for banks, leading to revenue loss and increased customer acquisition costs. Understanding why customers leave—and predicting who is likely to leave—empowers banks to offer targeted incentives or support to retain valuable customers.

---

## 💡 Solution
This project offers a robust solution to predict churn by:
- Building a **[Decision Tree Classifier & Random Forest Classifier]** to accurately identify potential churners.
- Analyzing key features that influence churn for actionable business insights.
- Generating churn probability scores to prioritize customer interventions.

---

## 🛠️ Technical Stack
- **Language**: Python **[3.13.1]**
- **Core Libraries**:  
  - `pandas` – Data manipulation  
  - `numpy` – Numerical operations  
  - `scikit-learn` – Model building & evaluation  
  - `matplotlib`, `seaborn` – Data visualization  
  - `Other libraries: xgboost, lightgbm, tensorflow, keras, imbalanced-learn, etc.`

---

## 📊 Dataset
- **Dataset Name**: *[Bank Customer Churn Dataset(kaggle)]*  
- **Shape**: *[10,000 rows × 12 columns]*  
- **Source**: [Dataset Link]([https://www.kaggle.com/datasets/shubhamghimire/bank-customer-churn-dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset?resource=download))

---

## 🔍 Key Features
The dataset includes:
- `CreditScore`  
- `Country ` (France, Spain, Germany)  
- `Gender`  
- `Age`  
- `Tenure` (Years with the bank)  
- `Balance`  
- `NumOfProducts`  
- `Creditcard` (Credit card status)  
- `IsActiveMember`  
- `EstimatedSalary`

---

## ⚙️ Data Preprocessing
Key preprocessing steps:
- **Categorical Encoding**: One-Hot Encoding for `Country ` and `Gender`.
- **Feature Scaling**: Applied **StandardScaler** to normalize numeric values.
- **Class Imbalance Handling**: Used **SMOTE** to balance the 'Exited' class.

---

## 📈 Model Performance
Best model used: **[Decision Tree Classifier]**

| Metric             | Score         |
|--------------------|---------------|
| Accuracy           | **[86.50%]** |
| Precision (Churn)  | **[83.00%]** |
| Recall (Churn)     | **[44.00%]** |
| F1-Score (Churn)   | **[58.00%]** |
| AUC Score          | **[0.8520]**  |

The model effectively balances between identifying true churners and minimizing false positives.

---

## 📁 Project Structure
```
│── bank_churn_data.csv
│── Exploratory_Data_Analysis.ipynb
│── Model_Training_and_Evaluation.ipynb
│── data_preprocessing.py
│── model_training.py
│── predict.py
├── requirements.txt
├── README.md
```

- `data/`: Raw dataset  
- `notebooks/`: Jupyter notebooks for EDA and model development  
- `src/`: Modular scripts for data processing, training, and prediction  
- `requirements.txt`: Project dependencies  

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/mdpatel007/bank-churn-ml-model]
cd [bank-churn-ml-model]
```

### 2. Set up a virtual environment (recommended)
```bash
python -m venv venv
# Activate environment:
# For Windows:
venv\Scripts\activate
# For Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run model training
```bash
python src/model_training.py
```

### 5. Explore with Jupyter
```bash
jupyter notebook
```
Open `Model_Training_and_Evaluation.ipynb` to view and run all steps interactively.

### 6. Make Predictions
Use:
- `src/predict.py`
- Or run prediction cells in the notebook

---

## 🚀 Future Enhancements
- **Model Deployment**: Deploy via Flask or FastAPI  
- **Feature Engineering**: Introduce advanced or interaction-based features  
- **Explainability**: Add SHAP/LIME for model interpretability  
- **Hyperparameter Tuning**: Use Optuna or GridSearchCV  
- **Monitoring**: Implement drift detection and performance logging  

---

## 🙏 Acknowledgments
- **Dataset**: Provided by [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset?resource=download)  

---

## 🌐 Live Demo
👉 [Click here to try the model!](https://bank-churn-ml-model-j2mwcappqjf9qpzlugaegsr.streamlit.app/)







