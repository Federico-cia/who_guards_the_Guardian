import argparse
import math
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from rank_bm25 import BM25Okapi


# Given a text, lowercase it and return a list of words (tokens)
def bm25_tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)

# Tokenize one batch of texts
def tokenize_batch(text_batch):
    return [bm25_tokenize(text) for text in text_batch]

# Tokenize all chunk texts, optionally using multiple workers in parallel
def parallel_tokenize(texts, n_workers=1, batch_size=2000):
    if n_workers <= 1:
        return [bm25_tokenize(text) for text in texts]

    # Split texts into batches
    batches = [
        texts[i:i + batch_size]
        for i in range(0, len(texts), batch_size)
    ]

    # Process batches in parallel
    tokenized = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in executor.map(tokenize_batch, batches):
            tokenized.extend(result)

    return tokenized


def main():
    # Read command-line arguments passed by the .sh script
    parser = argparse.ArgumentParser(description="Build BM25 assets in parallel on HPC.")
    parser.add_argument("--input", type=str, default="chunked_df_small.parquet")
    parser.add_argument("--output_parquet", type=str, default="chunked_df_small_bm25.parquet")
    parser.add_argument("--output_bm25", type=str, default="bm25_index_small.pkl")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2000)
    args = parser.parse_args()

    # Check that the input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load chunk-level dataframe
    print("Loading parquet...")
    chunked_df = pd.read_parquet(args.input)

    # Check that the text column needed for BM25 exists
    if "chunk_text" not in chunked_df.columns:
        print("ERROR: column 'chunk_text' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Rows loaded: {len(chunked_df)}")
    print(f"Using workers: {args.workers}")
    print("Tokenizing chunk_text in parallel...")

    # Convert chunk texts to a list and tokenize them
    texts = chunked_df["chunk_text"].astype(str).tolist()
    tokenized = parallel_tokenize(
        texts=texts,
        n_workers=args.workers,
        batch_size=args.batch_size
    )

    # Store BM25 tokens inside the dataframe
    chunked_df = chunked_df.copy()
    chunked_df["bm25_tokens"] = tokenized

    # Build the BM25 index from the tokenized chunks
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)

    # Save dataframe with BM25 tokens
    print("Saving enriched parquet...")
    chunked_df.to_parquet(args.output_parquet, index=False)

    # Save BM25 object for later retrieval in the notebook
    print("Saving BM25 object...")
    with open(args.output_bm25, "wb") as f:
        pickle.dump(bm25, f)

    print("Done.")
    print(f"Saved parquet: {args.output_parquet}")
    print(f"Saved BM25 index: {args.output_bm25}")


if __name__ == "__main__":
    main()