import pandas as pd
import os

RAW_DATA_PATH = "data/raw/accepted_2007_to_2018Q4.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_loan_data.csv"

def load_and_clean_data():
    print("🔄 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)

    print("📉 Filtering rows and selecting features...")
    # Drop rows with missing loan status or description
    df = df[~df['loan_status'].isna()]
    df = df[~df['desc'].isna()]  # We'll use this field for GenAI

    # Binary target: 1 if loan is charged off, 0 otherwise
    df['default_ind'] = df['loan_status'].apply(
        lambda x: 1 if "Charged Off" in x else 0
    )

    # Select features for modeling
    selected_cols = [
        'loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc',
        'purpose', 'addr_state', 'dti', 'fico_range_high', 'desc', 'default_ind'
    ]
    df = df[selected_cols]

    # Drop rows with missing key data
    df = df.dropna(subset=['loan_amnt', 'annual_inc', 'dti', 'fico_range_high'])

    # Simplify term
    df['term'] = df['term'].str.extract('(\d+)').astype(int)

    # Clean emp_length
    df['emp_length'] = df['emp_length'].replace({
    '10+ years': 10,
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 year': 1,
    '< 1 year': 0,
    'n/a': None
}).astype(float)


    print("💾 Saving cleaned dataset...")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Cleaned data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    load_and_clean_data()
