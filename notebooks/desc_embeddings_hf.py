import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import os

# Load borrower descriptions
df = pd.read_csv("data/processed/borrower_descs.csv")
print(f"✅ Loaded {len(df)} descriptions")

# Optional: Sample if data is too large
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Load embedding model
print("🧠 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast + good quality

# Generate embeddings
print("🔁 Generating embeddings...")
embeddings = model.encode(
    df['desc'].tolist(),
    show_progress_bar=True,
    batch_size=64
)

# Save as DataFrame
embedding_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
embedding_df['default_ind'] = df['default_ind'].values

# Combine and save
output_path = "data/processed/desc_embeddings_hf.csv"
embedding_df.to_csv(output_path, index=False)
print(f"✅ Embeddings saved to {output_path}")
