
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from xgboost import XGBClassifier
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("models/final_genai_structured_model.json")
    return model

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def classify_risk(desc):
    desc = desc.lower()
    if any(kw in desc for kw in ['unemployed', 'medical', 'late', 'bankrupt', 'eviction', 'behind']):
        return "High"
    elif any(kw in desc for kw in ['credit card', 'consolidate', 'pay off']):
        return "Medium"
    elif any(kw in desc for kw in ['business', 'home improvement', 'wedding', 'auto']):
        return "Low"
    else:
        return "Unknown"

model = load_model()
embedder = load_embedder()

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("🧠 Credit Risk Predictor")

with st.form("credit_form"):
    desc = st.text_area("📄 Borrower Explanation", placeholder="Why are you applying for this loan?")

    st.subheader("📊 Structured Loan Info")

    loan_amnt = st.number_input("💵 Loan Amount", value=10000, step=500)
    annual_inc = st.number_input("📈 Annual Income", value=60000)
    dti = st.number_input("🧮 Debt-to-Income Ratio (DTI)", value=15.0)
    fico = st.slider("🔐 FICO Score (High Range)", min_value=600, max_value=850, value=700)

    term = st.selectbox("⏱️ Term (months)", options=[36, 60])
    emp_length = st.selectbox("👷 Employment Length (years)", options=list(range(0, 11)))
    home_ownership = st.selectbox("🏠 Home Ownership", options=["RENT", "OWN", "MORTGAGE", "OTHER", "NONE"])
    purpose = st.selectbox("🎯 Loan Purpose", options=[
        "credit_card", "debt_consolidation", "home_improvement", "small_business", "medical",
        "vacation", "wedding", "major_purchase", "other", "house", "moving", "educational", "renewable_energy"
    ])
    addr_state = st.selectbox("📍 State", options=[
        "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA", "IN",
        "TN", "MO", "MD", "WI", "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "IA", "MS", "AR",
        "KS", "UT", "NV", "NM", "NE", "WV", "ID", "HI", "NH", "ME", "RI", "MT", "DE", "SD", "ND", "VT",
        "DC", "WY"
    ])

    submitted = st.form_submit_button("Submit")

if submitted:
    st.write("📋 You entered:")
    st.write(desc)

    risk = classify_risk(desc)
    st.write(f"🧠 Risk Tag: `{risk}`")

    embedding = embedder.encode([desc])[0]

    data_point = {
        "loan_amnt": loan_amnt,
        "term": term,
        "emp_length": emp_length,
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_range_high": fico,
        f"home_ownership_{home_ownership}": 1,
        f"purpose_{purpose}": 1,
        f"addr_state_{addr_state}": 1,
        f"rule_based_risk_{risk}": 1
    }

    expected_cols = joblib.load("models/final_feature_names.pkl")

    input_row = []
    for col in expected_cols:
        if col.startswith("emb_"):
            idx = int(col.split("_")[1])
            input_row.append(embedding[idx])
        else:
            input_row.append(data_point.get(col, 0))

    input_df = pd.DataFrame([input_row], columns=expected_cols)
    prob = model.predict_proba(input_df)[0][1]
    st.metric("🎯 Default Risk Probability", f"{prob:.2%}")
