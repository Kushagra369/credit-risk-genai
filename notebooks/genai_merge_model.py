import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

# Load GenAI embeddings + rule tags
emb = pd.read_csv("data/processed/desc_embeddings_hf.csv")
risk = pd.read_csv("data/processed/desc_risk_rule_based.csv")[['rule_based_risk']]

# Merge risk into embeddings
df_genai = pd.concat([emb, risk], axis=1)
print(f"✅ Merged GenAI features: {df_genai.shape}")

# One-hot encode risk tag
df_genai = pd.get_dummies(df_genai, columns=['rule_based_risk'], dummy_na=True)

# Final features and label
X = df_genai.drop(columns=["default_ind"])
y = df_genai["default_ind"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Train model
print("\n🚀 Training XGBoost with GenAI features...")
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
)
model.fit(X_train, y_train)

# Predict + evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n📈 ROC AUC Score:", roc_auc_score(y_test, y_prob))
