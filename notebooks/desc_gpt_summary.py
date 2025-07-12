import os
import pandas as pd
import time
from tqdm import tqdm
import openai
from openai import OpenAI

# Load borrower description data
df = pd.read_csv("data/processed/borrower_descs.csv").dropna(subset=['desc'])
df = df.sample(n=100, random_state=42).reset_index(drop=True)

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

summaries = []
tags = []

def call_openai_gpt(desc_text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You're a credit analyst reviewing borrower applications."
                },
                {
                    "role": "user",
                    "content": f"Summarize this borrower's explanation:\n\n'{desc_text}'\n\nAlso, classify the risk level as High, Medium, or Low based on repayment likelihood."
                }
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Process borrower descriptions
print("⚡ Calling OpenAI API...")
for desc in tqdm(df['desc']):
    result = call_openai_gpt(desc)
    time.sleep(1)
    if result:
        if "Risk level:" in result:
            summary, risk = result.split("Risk level:")
        else:
            summary, risk = result, "Unknown"
        summaries.append(summary.strip())
        tags.append(risk.strip())
    else:
        summaries.append("Error")
        tags.append("Unknown")

# Save result
df['gpt_summary'] = summaries
df['gpt_risk_tag'] = tags
df.to_csv("data/processed/desc_gpt_summaries.csv", index=False)
print("✅ Saved to data/processed/desc_gpt_summaries.csv")
