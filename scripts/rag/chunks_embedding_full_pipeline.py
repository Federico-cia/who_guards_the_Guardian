import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


# Split one article into overlapping chunks of at most max_tokens tokens
def chunk_article(text, title, tokenizer, max_tokens=512, overlap_ratio=0.15):
    # Split article body into sentences
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    chunks = []
    i = 0

    # Build chunks by adding consecutive sentences until the token limit is reached
    while i < len(sentences):
        chunk_sents = []
        chunk_text = ""
        j = i

        while j < len(sentences):
            candidate_body = (chunk_text + " " + sentences[j]).strip()
            candidate_full = f"{title}. {candidate_body}" if title else candidate_body

            # Check wether adding the next sentence keeps the chunk within max_tokens
            if len(tokenizer.encode(candidate_full, add_special_tokens=False)) <= max_tokens:
                chunk_sents.append(sentences[j])
                chunk_text = candidate_body
                j += 1
            else:
                break

        # If a single sentence is longer than max_tokens, truncate it to avoid getting stuck
        if not chunk_sents:
            available_tokens = max_tokens - len(tokenizer.encode(f"{title}. ", add_special_tokens=False)) if title else max_tokens
            available_tokens = max(1, available_tokens)
            token_ids = tokenizer.encode(sentences[i], add_special_tokens=False)[:available_tokens]
            chunk_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            j = i + 1

        # Store chunk text and metadata
        full_text = f"{title}. {chunk_text}" if title else chunk_text
        chunks.append({
            "chunk_text": full_text,
            "chunk_tokens": len(tokenizer.encode(full_text, add_special_tokens=False)),
            "start_sentence_idx": i,
            "end_sentence_idx": j - 1,
        })

        if j >= len(sentences):
            break

        # Reuse the last part of the current chunk as overlap for the next chunk
        overlap_target = int(max_tokens * overlap_ratio)
        overlap_tokens = 0
        overlap_sentences = 0
        for s in reversed(chunk_sents):
            overlap_tokens += len(tokenizer.encode(s, add_special_tokens=False))
            overlap_sentences += 1
            if overlap_tokens >= overlap_target:
                break

        # Move forward by at least one sentence
        i = max(i + 1, j - overlap_sentences)

    return chunks


def main():
    # Read file paths and hyperparameters from command-line arguments passed by the .sh script
    parser = argparse.ArgumentParser(description="Create chunk-level Guardian data and save embeddings separately.")
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--output_light_parquet", required=True)
    parser.add_argument("--model_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--overlap_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    input_parquet = Path(args.input_parquet)
    output_npy = Path(args.output_npy)
    output_light_parquet = Path(args.output_light_parquet)

    # Create output folders if they do not exist
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_light_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Load the cleaned Guardian article dataset
    print(f"Reading input parquet: {input_parquet}", flush=True)
    articles_clean = pd.read_parquet(input_parquet)

    # Check that the required text columns are present
    if "bodyContent" not in articles_clean.columns or "webTitle" not in articles_clean.columns:
        raise ValueError("Input file must contain 'bodyContent' and 'webTitle' columns.")

    articles_clean = articles_clean.copy()

    # Keep the original article index so each chunk can be traced back to its source article
    articles_clean["orig_index"] = articles_clean.index

    # Drop columns not needed for the RAG chunk-level dataset
    articles_clean = articles_clean.drop(
        columns=["month", "month_start", "bodyContent_lemma", "body_sent_lemma_tokens"],
        errors="ignore",
    )

    # Load tokenizer for token counting and BGE model for embedding computation
    print(f"Loading tokenizer and embedding model: {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    embedding_model = SentenceTransformer(args.model_name)

    # Create one row per article chunk
    print("Creating article chunks...", flush=True)
    rows = []
    for row_num, row in articles_clean.iterrows():
        body = "" if pd.isna(row["bodyContent"]) else re.sub(r"\s+", " ", str(row["bodyContent"])).strip()
        title = "" if pd.isna(row["webTitle"]) else re.sub(r"\s+", " ", str(row["webTitle"])).strip()

        # Skip articles with empty body
        if not body:
            continue

        # Split the current article into overlapping chunks
        article_chunks = chunk_article(
            text=body,
            title=title,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            overlap_ratio=args.overlap_ratio,
        )

        # Store chunk text and metadata
        for chunk_num, ch in enumerate(article_chunks):
            rows.append({
                "orig_index": row["orig_index"],
                "webTitle": row["webTitle"],
                "Date": row["Date"] if "Date" in articles_clean.columns else None,
                "sectionName": row["sectionName"] if "sectionName" in articles_clean.columns else None,
                "year": row["year"] if "year" in articles_clean.columns else None,
                "year_month": row["year_month"] if "year_month" in articles_clean.columns else None,
                "chunk_id": f"{row['orig_index']}_{chunk_num}",
                "chunk_num": chunk_num,
                "chunk_text": ch["chunk_text"],
                "chunk_tokens": ch["chunk_tokens"],
                "start_sentence_idx": ch["start_sentence_idx"],
                "end_sentence_idx": ch["end_sentence_idx"],
            })

        if (row_num + 1) % 5000 == 0:
            print(f"Processed {row_num + 1} articles...", flush=True)

    # Convert the list of chunk dictionaries into a dataframe
    chunked_df = pd.DataFrame(rows)
    
    print(f"Created chunks: {len(chunked_df)}", flush=True)
    print(f"Chunks above max token limit: {(chunked_df['chunk_tokens'] > args.max_tokens).sum()}", flush=True)

    # Save chunk text and metadata without embeddings
    print(f"Saving light parquet without embeddings: {output_light_parquet}", flush=True)
    chunked_df.to_parquet(output_light_parquet, index=False)

    # Get the embedding dimension of the selected model
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    if embedding_dim is None:
        sample_embedding = embedding_model.encode([chunked_df["chunk_text"].iloc[0]], convert_to_numpy=True)
        embedding_dim = sample_embedding.shape[1]

    # Create a memory-mapped .npy file to write embeddings without keeping all of them in RAM
    print(f"Creating memory-mapped embedding matrix: {output_npy}", flush=True)
    print(f"Embedding matrix shape: ({len(chunked_df)}, {embedding_dim})", flush=True)
    embeddings = open_memmap(
        output_npy,
        mode="w+",
        dtype=np.float32,
        shape=(len(chunked_df), embedding_dim),
    )

    # Compute chunk embeddings in batches and save them directly into the .npy file
    print("Computing and saving chunk embeddings...", flush=True)
    for start in range(0, len(chunked_df), args.batch_size):
        end = min(start + args.batch_size, len(chunked_df))
        batch_texts = chunked_df["chunk_text"].iloc[start:end].tolist()

        # Convert each chunk text into a dense vector embedding using BGE
        batch_embeddings = embedding_model.encode(
            batch_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Save the batch embeddings into the correct rows of the memory-mapped matrix
        embeddings[start:end] = batch_embeddings.astype(np.float32)

        if end % 50000 == 0 or end == len(chunked_df):
            print(f"Saved embeddings for {end}/{len(chunked_df)} chunks", flush=True)

    # Force embeddings to be written to disk and free memory
    embeddings.flush()
    del embeddings
    gc.collect()

    print("Done.", flush=True)
    print(f"Embeddings saved to: {output_npy}", flush=True)
    print(f"Light parquet saved to: {output_light_parquet}", flush=True)


if __name__ == "__main__":
    main()
