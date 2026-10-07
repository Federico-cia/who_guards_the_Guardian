import argparse
import os
from datasets import load_dataset, DatasetDict
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast


def load_guardian_dataset(dataset_name=None, train_file=None):
    if dataset_name is not None:
        return load_dataset(dataset_name)

    if train_file is None:
        raise ValueError("Provide either --dataset_name or --train_file.")

    ext = train_file.split(".")[-1].lower()

    if ext == "csv":
        return load_dataset("csv", data_files=train_file)
    if ext in ["parquet", "pq"]:
        return load_dataset("parquet", data_files=train_file)
    if ext == "json":
        return load_dataset("json", data_files=train_file)

    raise ValueError("Supported formats: csv, parquet, json")


def get_train_split(dataset):
    if isinstance(dataset, DatasetDict):
        return dataset["train"]
    return dataset


def batch_iterator(dataset, text_column, batch_size=1000):
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size][text_column]
        yield [
            text
            for text in batch
            if isinstance(text, str) and len(text.strip()) > 0
        ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--text_column", default="bodyContent")
    parser.add_argument("--output_dir", default="./guardian_electra_tokenizer")
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--lowercase", action="store_true")

    args = parser.parse_args()

    dataset = load_guardian_dataset(
        dataset_name=args.dataset_name,
        train_file=args.train_file,
    )

    train_dataset = get_train_split(dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=args.lowercase,
    )

    tokenizer.train_from_iterator(
        batch_iterator(train_dataset, args.text_column),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.save_model(args.output_dir)

    fast_tokenizer = BertTokenizerFast.from_pretrained(
        args.output_dir,
        do_lower_case=args.lowercase,
    )

    fast_tokenizer.save_pretrained(args.output_dir)

    print(f"Tokenizer saved to: {args.output_dir}")
    print(f"Vocabulary size: {len(fast_tokenizer)}")


if __name__ == "__main__":
    main()
