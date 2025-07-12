import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Settings
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

# Load dataset
df = pd.read_csv("data/processed/cleaned_loan_data.csv")

print(f"✅ Loaded data with shape: {df.shape}")
print(df.head())

print("\n🔍 Data Types:")
print(df.dtypes)

print("\n📉 Missing Values:")
print(df.isnull().sum())

print("\n📊 Default Class Distribution:")
print(df["default_ind"].value_counts(normalize=True).round(3))

sns.countplot(x="default_ind", data=df)
plt.title("Loan Default Distribution (0 = Not Defaulted, 1 = Defaulted)")

# Save to a file
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/default_distribution.png")
plt.clf()
 # Loan Amount
sns.histplot(df['loan_amnt'], kde=True)
plt.title("Loan Amount Distribution")
plt.savefig("plots/loan_amount.png")
plt.clf()

# Annual Income
sns.histplot(df['annual_inc'], kde=True)
plt.title("Annual Income Distribution")
plt.savefig("plots/annual_income.png")
plt.clf()

# DTI (Debt-to-Income)
sns.histplot(df['dti'], kde=True)
plt.title("Debt-to-Income Ratio")
plt.savefig("plots/dti_distribution.png")
plt.clf()

# FICO Range
sns.histplot(df['fico_range_high'], kde=True)
plt.title("FICO Score (High End)")
plt.savefig("plots/fico_score.png")
plt.clf()

print("\n📝 Sample Borrower Descriptions:")
print(df['desc'].dropna().sample(5, random_state=42).values)
