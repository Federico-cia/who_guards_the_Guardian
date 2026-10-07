import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default="./guardian_electra_small_3ep_1619/discriminator",
        help="Path to the pretrained ELECTRA discriminator.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./guardian_electra_mnli",
        help="Where to save the fine-tuned NLI model.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--mnli_train_file",
        type=str,
        default="mnli_train.parquet",
        help="Path to the local MultiNLI training parquet file.",
    )
    parser.add_argument(
        "--mnli_validation_matched_file",
        type=str,
        default="mnli_validation_matched.parquet",
        help="Path to the matched MultiNLI validation parquet file.",
    )
    parser.add_argument(
        "--mnli_validation_mismatched_file",
        type=str,
        default="mnli_validation_mismatched.parquet",
        help="Path to the mismatched MultiNLI validation parquet file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading MultiNLI dataset...")
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": args.mnli_train_file,
            "validation_matched": args.mnli_validation_matched_file,
            "validation_mismatched": args.mnli_validation_mismatched_file,
        },
    )

    print(dataset)

    label_names = dataset["train"].features["label"].names
    print("Label names:", label_names)

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=args.max_seq_length,
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=args.num_proc,
    )

    columns_to_remove = [
        col
        for col in tokenized_dataset["train"].column_names
        if col not in ["input_ids", "attention_mask", "token_type_ids", "label"]
    ]

    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

    print("Loading pretrained ELECTRA discriminator for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        bf16=True,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation_matched"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting MNLI fine-tuning...")
    trainer.train()

    print("Evaluating on validation_matched...")
    matched_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["validation_matched"],
        metric_key_prefix="matched",
    )
    print(matched_metrics)

    print("Evaluating on validation_mismatched...")
    mismatched_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["validation_mismatched"],
        metric_key_prefix="mismatched",
    )
    print(mismatched_metrics)

    print("Saving final fine-tuned NLI model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Fine-tuned MNLI model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
