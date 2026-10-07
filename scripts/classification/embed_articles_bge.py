import os
import math
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Build the text that will be passed to the embedding model
def build_text(title, body, max_chars=4000):
    title = "" if pd.isna(title) else str(title)
    body = "" if pd.isna(body) else str(body)
    text = f"Title: {title}\nBody: {body}".strip()
    text = " ".join(text.split())
    return text[:max_chars]


def main():
    parser = argparse.ArgumentParser(description="Embed Guardian articles with BGE base.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--model_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--title_col", default="webTitle")
    parser.add_argument("--body_col", default="bodyContent")
    parser.add_argument("--max_chars", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--end_row", type=int, default=None)
    parser.add_argument("--normalize_embeddings", action="store_true")
    args = parser.parse_args()

    # create output folders
    os.makedirs(args.output_dir, exist_ok=True)
    chunks_dir = os.path.join(args.output_dir, "embedding_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Load input dataset
    print("Loading CSV...")
    df = pd.read_csv(args.input_csv)

    # If no end row is specified, process the full dataset
    if args.end_row is None:
        args.end_row = len(df)

    # Select the rows to process
    df = df.iloc[args.start_row:args.end_row].copy().reset_index(drop=True)
    print(f"Rows to process: {len(df)}")

    # Create the final text used for article-level embedding: title + body
    print("Building text field...")
    df["text_for_embedding"] = [
        build_text(t, b, max_chars=args.max_chars)
        for t, b in zip(df[args.title_col], df[args.body_col])
    ]

    # Load the SentenceTransforme model
    print("Loading model...")
    model = SentenceTransformer(args.model_name)
    print(f"Model device: {model.device}")
    print(f"Model max_seq_length: {model.max_seq_length}")

    total_rows = len(df)
    texts = df["text_for_embedding"].tolist()

    # Prepare temporary containers for chunked saving
    buffer_embeddings = []
    buffer_start = 0
    processed = 0
    chunk_manifest = []

    # Compute embeddings in batches
    print("Starting embedding...")
    for batch_start in range(0, total_rows, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_rows)
        batch_texts = texts[batch_start:batch_end]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=args.normalize_embeddings,
        )

        buffer_embeddings.append(batch_embeddings)
        processed = batch_end

        # Save embeddings every save_every rows to avoid keeping everything in RAM
        if (processed - buffer_start) >= args.save_every or processed == total_rows:
            chunk = np.vstack(buffer_embeddings)
            global_start = args.start_row + buffer_start
            global_end = args.start_row + processed
            chunk_name = f"embeddings_{global_start}_{global_end}.npy"
            chunk_path = os.path.join(chunks_dir, chunk_name)
            np.save(chunk_path, chunk)
            print(f"Saved {chunk_name} with shape {chunk.shape}")

            # Store information about the saved chunk
            chunk_manifest.append({
                "chunk_file": chunk_name,
                "global_start_row": global_start,
                "global_end_row": global_end,
                "n_rows": int(chunk.shape[0]),
                "embedding_dim": int(chunk.shape[1]),
            })

            buffer_embeddings = []
            buffer_start = processed

    # Saving full metadata with embedding row index
    print("Saving metadata...")
    df["embedding_index"] = np.arange(args.start_row, args.start_row + total_rows)

    metadata_cols = [c for c in df.columns if c != "text_for_embedding"] + ["text_for_embedding"]
    metadata_path = os.path.join(args.output_dir, "articles_with_embedding_index.parquet")
    manifest_path = os.path.join(args.output_dir, "embedding_manifest.csv")

    df[metadata_cols].to_parquet(metadata_path, index=False)
    pd.DataFrame(chunk_manifest).to_csv(manifest_path, index=False)

    # Reconstruct one full embedding matrix from saved chunks
    print("Saving full embedding matrix...")
    full_embeddings_path = os.path.join(args.output_dir, "embeddings_full.npy")
    if len(chunk_manifest) == 0:
        raise ValueError("No embedding chunks were created.")

    embedding_dim = int(chunk_manifest[0]["embedding_dim"])
    full_embeddings = np.lib.format.open_memmap(
        full_embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, embedding_dim),
    )

    cursor = 0
    for item in chunk_manifest:
        chunk_path = os.path.join(chunks_dir, item["chunk_file"])
        chunk = np.load(chunk_path, mmap_mode="r")
        n_rows = chunk.shape[0]
        full_embeddings[cursor:cursor + n_rows] = chunk
        cursor += n_rows

    full_embeddings.flush()
    del full_embeddings

    # Save reduced metadata used directly in the notebook
    print("Saving reduced metadata for the notebook...")
    small_metadata_cols = [
        "webTitle",
        "sectionName",
        "crime_score",
        "immigration_score",
        "zs_crime_score",
        "zs_immigration_score",
        "embedding_index",
    ]
    missing_small_cols = [c for c in small_metadata_cols if c not in df.columns]
    if missing_small_cols:
        raise ValueError(f"Missing columns for articles_metadata_small.parquet: {missing_small_cols}")

    small_metadata_path = os.path.join(args.output_dir, "articles_metadata_small.parquet")
    df[small_metadata_cols].to_parquet(small_metadata_path, index=False)

    print("Done.")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Chunk manifest saved to: {manifest_path}")
    print(f"Full embeddings saved to: {full_embeddings_path}")
    print(f"Reduced metadata saved to: {small_metadata_path}")


if __name__ == "__main__":
    main()
