import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/processed/cleaned_loan_data.csv")
print(f"✅ Data loaded: {df.shape}")

# Drop unused columns
drop_cols = ['desc']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# One-hot encode
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop(columns=['default_ind'])
y = df_encoded['default_ind']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"🧪 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train XGBoost model
print("\n🚀 Training XGBoost...")
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # handle imbalance
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n📈 ROC AUC Score:", roc_auc_score(y_test, y_prob))
