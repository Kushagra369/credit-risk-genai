import pandas as pd

# Load descriptions
df = pd.read_csv("data/processed/borrower_descs.csv").dropna(subset=['desc'])
df = df.sample(n=1000, random_state=42).reset_index(drop=True)

# Define risk rules (basic keyword-based)
def classify_risk(desc):
    desc = desc.lower()

    # High risk indicators
    if any(kw in desc for kw in ['unemployed', 'medical', 'late', 'bankrupt', 'eviction', 'behind']):
        return "High"
    
    # Medium risk indicators
    elif any(kw in desc for kw in ['credit card', 'consolidate', 'pay off']):
        return "Medium"

    # Low risk indicators
    elif any(kw in desc for kw in ['business', 'home improvement', 'wedding', 'auto']):
        return "Low"
    
    # Unknown if no match
    else:
        return "Unknown"

# Apply classification
df['rule_based_risk'] = df['desc'].apply(classify_risk)

# Save output
df.to_csv("data/processed/desc_risk_rule_based.csv", index=False)
print("✅ Saved: data/processed/desc_risk_rule_based.csv")
print(df[['desc', 'rule_based_risk']].head())
