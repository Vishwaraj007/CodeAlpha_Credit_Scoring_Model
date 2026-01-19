import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/credit_risk_dataset.csv")

# Encode categorical columns
encoder = LabelEncoder()
categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Features & target
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "model/credit_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model trained successfully")

