import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

# Load GenAI embeddings + rule tags
emb = pd.read_csv("data/processed/desc_embeddings_hf.csv")
risk = pd.read_csv("data/processed/desc_risk_rule_based.csv")[['rule_based_risk']]

# Merge risk into embeddings
df_genai = pd.concat([emb, risk], axis=1)
print(f"✅ Merged GenAI features: {df_genai.shape}")

# One-hot encode rule-based risk tag
df_genai = pd.get_dummies(df_genai, columns=['rule_based_risk'], dummy_na=True)

# Define features and label
X = df_genai.drop(columns=["default_ind"])
y = df_genai["default_ind"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Apply oversampling
print("🔁 Applying RandomOverSampler...")
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
print(f"✅ Resampled training shape: {X_train_resampled.shape}")

# Train XGBoost model
print("\n🚀 Training XGBoost with GenAI + Oversampling...")
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_resampled, y_train_resampled)

# Predict + Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n📈 ROC AUC Score:", roc_auc_score(y_test, y_prob))
