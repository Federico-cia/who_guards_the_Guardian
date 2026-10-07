import pandas as pd
import torch
from transformers import pipeline

INPUT_FILE = "articles_clean_with_both_lemma_columns.parquet"
OUTPUT_FILE = "articles_clean_with_lemmas_and_zs_scores.parquet"

# Use only first 600 characters of the article body
MAX_BODY_CHARS = 600
BATCH_SIZE = 16

# Load imput file
print("Loading data...", flush=True)
df = pd.read_parquet(INPUT_FILE)

print("Rows:", len(df), flush=True)
print("Columns:", list(df.columns), flush=True)

# Build the text passed to the zero-shot classifier: title + beginning of article
print("Building zs_text...", flush=True)
df["zs_text"] = (
    df["webTitle"].fillna("").astype(str) + ". " +
    df["bodyContent"].fillna("").astype(str).str.slice(0, MAX_BODY_CHARS)
)

# Use GPU if available, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}", flush=True)


print("Loading zero-shot model...", flush=True)
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
    device=device
)

# Define the candidate labels and the hypothesis template
labels = ["crime", "immigration"]
hypothesis_template = "This article is about {}."

# Create empty lists to store scores
crime_scores = []
immigration_scores = []

texts = df["zs_text"].tolist()

# Run zero.shot classification in batches
print("Starting zero-shot inference...", flush=True)
for start in range(0, len(texts), BATCH_SIZE):
    batch = texts[start:start + BATCH_SIZE]

    # Classifiy each article in batch
    results = classifier(
        batch,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        multi_label=True
    )

    # Ensure results is always a list, even if batch contains only one element
    if isinstance(results, dict):
        results = [results]

    # Extract crime and immigration scores for each article
    for res in results:
        score_map = dict(zip(res["labels"], res["scores"]))
        crime_scores.append(score_map.get("crime", 0.0))
        immigration_scores.append(score_map.get("immigration", 0.0))

    # Print progress
    done = min(start + BATCH_SIZE, len(texts))
    if done % 100 == 0 or done == len(texts):
        print(f"Processed {done} / {len(texts)} articles...", flush=True)

# Add scores to the dataframe
df["zs_crime_score"] = crime_scores
df["zs_immigration_score"] = immigration_scores

# Save final dataframe
print("Saving output...", flush=True)
df.to_parquet(OUTPUT_FILE, index=False)

print(f"Done. Saved to {OUTPUT_FILE}", flush=True)
