import pandas as pd

# Load original CSV (may be large!)
raw_path = "data/raw/accepted_2007_to_2018Q4.csv"

print("🔄 Loading raw dataset (this might take a minute)...")
df = pd.read_csv(raw_path, usecols=['id', 'desc', 'loan_amnt', 'loan_status'])

# Drop NaNs
df = df.dropna(subset=['desc'])

# Optional: Filter only Charged Off or Fully Paid
df = df[df['loan_status'].isin(['Charged Off', 'Fully Paid'])]

# Map target
df['default_ind'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

# Save text + target only
df[['id', 'desc', 'default_ind']].to_csv("data/processed/borrower_descs.csv", index=False)
print("✅ Saved to data/processed/borrower_descs.csv")
