# 💳 Credit Scoring Model

**Project Name:** CodeAlpha Credit Scoring Model  
**Author:** Vishwaraj  
**Technology Stack:** Python, Scikit-Learn, Streamlit, Pandas, NumPy  

---

## 📌 Objective
Predict an individual's **creditworthiness** using past financial and personal data.  
This helps lenders make informed decisions by assessing loan default risk.

---

## 🛠️ Features

- **Machine Learning Model:**  
  - Random Forest Classifier (can be replaced with Logistic Regression / Decision Tree)
  - Predicts credit risk based on financial and personal attributes

- **Streamlit Web App UI:**  
  - Interactive input fields (sliders, text boxes, drop-downs)  
  - Instant prediction of creditworthiness

- **Feature Engineering:**  
  - Encodes categorical variables
  - Scales numerical features
  - Handles real dataset from Kaggle

- **Evaluation Ready:**  
  - Model can be extended to include metrics like **Accuracy, F1-Score, ROC-AUC**

---

## 📊 Dataset

- Key features:
  - `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`
  - `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`
  - `loan_status` (target), `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`

---

## 💻 Project Structure

credit_scoring/
│
├── data/
│ └── credit_risk_dataset.csv # Kaggle dataset
│
├── model/
│ ├── credit_model.pkl # Trained ML model
│ └── scaler.pkl # Feature scaler
│
├── train_model.py # Script to train the model
├── app.py # Streamlit web app
├── requirements.txt # Project dependencies
└── README.md # Project description


## 🚀 How to Run

### 1. Clone the repository
git clone https://github.com/Vishwaraj007/CodeAlpha_Credit_Scoring_Model.git

### 2. Setup Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### 3. Train the Model

python train_model.py


### 4. Run the Streamlit UI
streamlit run app.py

Open browser at http://localhost:8501 and interact with the app.

🎯 Usage
Enter user details: age, income, employment, home ownership, loan intent, grade, loan amount, interest rate, credit history.

Click Predict
App displays:

✅ Low Credit Risk

❌ High Credit Risk

📂 Dependencies

pandas
numpy
scikit-learn
joblib
streamlit

🔧 Future Improvements
  - Add model evaluation metrics (Accuracy, F1-score, ROC-AUC)
  - Visualize feature importance
  - Deploy app online (Heroku / Streamlit Cloud)
  - Add user authentication for real-time use

📝 Notes
Dataset sourced from Kaggle
Virtual environment recommended for dependency management
Designed for internship/demo purposes

📌 Author
Vishwaraj
GitHub: https://github.com/Vishwaraj007
