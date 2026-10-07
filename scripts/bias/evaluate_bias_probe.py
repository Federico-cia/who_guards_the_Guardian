#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned ELECTRA-MNLI model folder."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to bias-probe CSV."
    )

    parser.add_argument(
        "--output_predictions_csv",
        type=str,
        default="bias_probe_predictions.csv",
        help="Where to save row-level predictions."
    )

    parser.add_argument(
        "--output_metrics_csv",
        type=str,
        default="bias_probe_group_metrics.csv",
        help="Where to save group-level metrics."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for premise-hypothesis pairs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size."
    )

    return parser.parse_args()


def normalise_label_name(label):
    label = str(label).lower()

    if "entail" in label:
        return "entailment"
    if "neutral" in label:
        return "neutral"
    if "contrad" in label:
        return "contradiction"

    return label


def get_label_mapping(model):
    """
    Returns the index positions for entailment, neutral, contradiction
    according to the model config.

    This avoids assuming a fixed label order.
    """

    id2label = model.config.id2label

    normalised = {
        int(i): normalise_label_name(label)
        for i, label in id2label.items()
    }

    label_to_index = {label: idx for idx, label in normalised.items()}

    required = ["entailment", "neutral", "contradiction"]

    missing = [label for label in required if label not in label_to_index]

    if missing:
        print("WARNING: Could not infer label names cleanly from model.config.id2label.")
        print("model.config.id2label =", model.config.id2label)
        print("Falling back to assumed MultiNLI order:")
        print("0 = entailment, 1 = neutral, 2 = contradiction")

        label_to_index = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }

    return label_to_index


def predict_probabilities(df, tokenizer, model, device, label_to_index, max_length, batch_size):
    model.eval()

    all_neutral_probs = []
    all_entailment_probs = []
    all_contradiction_probs = []

    premises = df["premise"].astype(str).tolist()
    hypotheses = df["hypothesis"].astype(str).tolist()

    for start in tqdm(range(0, len(df), batch_size), desc="Predicting"):
        end = start + batch_size

        batch_premises = premises[start:end]
        batch_hypotheses = hypotheses[start:end]

        encoded = tokenizer(
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        probs = probs.detach().cpu()

        entailment_idx = label_to_index["entailment"]
        neutral_idx = label_to_index["neutral"]
        contradiction_idx = label_to_index["contradiction"]

        all_entailment_probs.extend(probs[:, entailment_idx].tolist())
        all_neutral_probs.extend(probs[:, neutral_idx].tolist())
        all_contradiction_probs.extend(probs[:, contradiction_idx].tolist())

    df["p_neutral"] = all_neutral_probs
    df["p_entailment"] = all_entailment_probs
    df["p_contradiction"] = all_contradiction_probs

    predicted_labels = []

    for _, row in df.iterrows():
        probs = {
            "neutral": row["p_neutral"],
            "entailment": row["p_entailment"],
            "contradiction": row["p_contradiction"],
        }
        predicted_labels.append(max(probs, key=probs.get))

    df["predicted_label"] = predicted_labels

    return df


def compute_group_metrics(df):
    """
    Computes requested group-level metrics.

    Metrics:
    1. Net Neutral
    2. Fraction Neutral
    3. Fraction Entailment
    4. Fraction Contradiction
    5. Threshold 0.5 Neutral
    6. Threshold 0.7 Neutral
    7. Threshold 0.5 Entailment
    8. Threshold 0.7 Entailment
    """

    rows = []

    for group, group_df in df.groupby("group"):
        n = len(group_df)

        net_neutral = group_df["p_neutral"].mean()

        fraction_neutral = (group_df["predicted_label"] == "neutral").mean()
        fraction_entailment = (group_df["predicted_label"] == "entailment").mean()
        fraction_contradiction = (group_df["predicted_label"] == "contradiction").mean()

        threshold_05_neutral = (group_df["p_neutral"] > 0.5).mean()
        threshold_07_neutral = (group_df["p_neutral"] > 0.7).mean()

        threshold_05_entailment = (group_df["p_entailment"] > 0.5).mean()
        threshold_07_entailment = (group_df["p_entailment"] > 0.7).mean()

        rows.append(
            {
                "group": group,
                "n_pairs": n,
                "net_neutral": net_neutral,
                "fraction_neutral": fraction_neutral,
                "fraction_entailment": fraction_entailment,
                "fraction_contradiction": fraction_contradiction,
                "threshold_0_5_neutral": threshold_05_neutral,
                "threshold_0_7_neutral": threshold_07_neutral,
                "threshold_0_5_entailment": threshold_05_entailment,
                "threshold_0_7_entailment": threshold_07_entailment,
            }
        )

    metrics_df = pd.DataFrame(rows)

    metrics_df = metrics_df.sort_values("group").reset_index(drop=True)

    return metrics_df


def main():
    args = parse_args()

    print("Loading input CSV...")
    df = pd.read_csv(args.input_csv)

    required_columns = ["premise", "hypothesis", "group", "x1", "x2"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Loaded {len(df)} sentence pairs.")
    print("Groups:")
    print(df["group"].value_counts())

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    label_to_index = get_label_mapping(model)
    print("Detected label mapping:")
    print(label_to_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.to(device)

    df_predictions = predict_probabilities(
        df=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        label_to_index=label_to_index,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print("Saving row-level predictions...")
    df_predictions.to_csv(args.output_predictions_csv, index=False)

    print("Computing group-level metrics...")
    metrics_df = compute_group_metrics(df_predictions)

    print(metrics_df)

    print("Saving group-level metrics...")
    metrics_df.to_csv(args.output_metrics_csv, index=False)

    print("Done.")
    print(f"Predictions saved to: {args.output_predictions_csv}")
    print(f"Metrics saved to: {args.output_metrics_csv}")


if __name__ == "__main__":
    main()

