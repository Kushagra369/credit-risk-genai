import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

# --- Load All Data ---

# Load structured data
structured = pd.read_csv("data/processed/cleaned_loan_data.csv")
structured = structured.sample(n=5000, random_state=42).reset_index(drop=True)

# Drop text field and target
desc_col = structured.pop("desc")
target = structured["default_ind"]
structured = structured.drop(columns=["default_ind"])

# One-hot encode categorical columns
structured = pd.get_dummies(structured, drop_first=True)

# Add back the target
structured["default_ind"] = target

# 2. HF embeddings
emb = pd.read_csv("data/processed/desc_embeddings_hf.csv")

# 3. Risk tags
risk = pd.read_csv("data/processed/desc_risk_rule_based.csv")[['rule_based_risk']]

# --- Merge All Together ---
df = pd.concat([structured, emb.drop(columns=["default_ind"]), risk], axis=1)
df = pd.get_dummies(df, columns=['rule_based_risk'], dummy_na=True)

print(f"✅ Final merged shape: {df.shape}")

# --- Train/Test Split ---
X = df.drop(columns=["default_ind"])
y = df["default_ind"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# --- Oversample ---
print("🔁 Applying RandomOverSampler...")
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
print(f"✅ Resampled shape: {X_train_res.shape}")

# --- Train Model ---
print("\n🚀 Training Final XGBoost Model...")
model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model.fit(X_train_res, y_train_res)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n📈 ROC AUC Score:", roc_auc_score(y_test, y_prob))

# --- Save model for Streamlit ---
model.save_model("models/final_genai_structured_model.json")
print("💾 Model saved to models/final_genai_structured_model.json")
