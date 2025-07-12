import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("data/processed/cleaned_loan_data.csv")
print(f"✅ Data loaded: {df.shape}")

# Drop 'desc' if it exists
drop_cols = ['desc']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# One-hot encode
df_encoded = pd.get_dummies(df, drop_first=True)

# Define X, y
X = df_encoded.drop(columns=["default_ind"])
y = df_encoded["default_ind"]

# Impute missing values (e.g., emp_length or unknown categories)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.3, random_state=42, stratify=y
)

print(f"🧪 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train model
print("\n🚀 Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n📈 ROC AUC Score:", roc_auc_score(y_test, y_prob))
