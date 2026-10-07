import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm


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


def ensure_train_validation(dataset, validation_split=0.02):
    if not isinstance(dataset, DatasetDict):
        split = dataset["train"].train_test_split(
            test_size=validation_split,
            seed=42,
        )
        return DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            }
        )

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(
            test_size=validation_split,
            seed=42,
        )
        return DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
            }
        )

    return dataset


def create_configs(tokenizer, max_seq_length):
    discriminator_config = ElectraConfig(
        vocab_size=len(tokenizer),
        embedding_size=128,
        hidden_size=256,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max(512, max_seq_length),
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=tokenizer.pad_token_id,
    )

    generator_config = ElectraConfig(
        vocab_size=len(tokenizer),
        embedding_size=128,
        hidden_size=64,
        num_hidden_layers=12,
        num_attention_heads=1,
        intermediate_size=256,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max(512, max_seq_length),
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=tokenizer.pad_token_id,
    )

    return generator_config, discriminator_config


def sample_from_generator(logits):
    probs = F.softmax(logits, dim=-1)
    sampled_tokens = torch.multinomial(
        probs.view(-1, probs.size(-1)),
        num_samples=1,
    ).view(logits.size()[:-1])
    return sampled_tokens


def make_electra_inputs(input_ids, generator_logits, generator_labels):
    sampled_tokens = sample_from_generator(generator_logits)

    masked_positions = generator_labels != -100
    original_tokens = generator_labels

    corrupted_input_ids = input_ids.clone()
    corrupted_input_ids[masked_positions] = sampled_tokens[masked_positions]

    discriminator_labels = torch.zeros_like(input_ids)
    replaced_positions = (
        masked_positions
        & (sampled_tokens != original_tokens)
    )
    discriminator_labels[replaced_positions] = 1

    return corrupted_input_ids, discriminator_labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--text_column", default="bodyContent")

    parser.add_argument("--tokenizer_dir", default="./guardian_electra_tokenizer")
    parser.add_argument("--output_dir", default="./guardian_electra_small")

    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--validation_split", type=float, default=0.02)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)

    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--generator_loss_weight", type=float, default=1.0)
    parser.add_argument("--discriminator_loss_weight", type=float, default=50.0)

    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_every_epoch", action="store_true")

    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_dir)

    dataset = load_guardian_dataset(
        dataset_name=args.dataset_name,
        train_file=args.train_file,
    )

    dataset = ensure_train_validation(
        dataset,
        validation_split=args.validation_split,
    )

    def clean_examples(batch):
        texts = batch[args.text_column]
        return {
            "text": [
                text
                if isinstance(text, str) and len(text.strip()) > 0
                else ""
                for text in texts
            ]
        }

    dataset = dataset.map(
        clean_examples,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_proc,
    )

    dataset = dataset.filter(
        lambda x: len(x["text"].strip()) > 0,
        num_proc=args.num_proc,
    )

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            return_special_tokens_mask=True,
            truncation=False,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=args.num_proc,
    )

    block_size = args.max_seq_length

    def group_texts(examples):
        concatenated = {
            key: sum(examples[key], [])
            for key in examples.keys()
        }

        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {
            key: [
                values[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }

        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=args.num_proc,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    train_loader = DataLoader(
        lm_dataset["train"],
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        lm_dataset["validation"],
        shuffle=False,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    generator_config, discriminator_config = create_configs(
        tokenizer,
        max_seq_length=args.max_seq_length,
    )

    generator = ElectraForMaskedLM(generator_config)
    discriminator = ElectraForPreTraining(discriminator_config)

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in list(generator.named_parameters())
                + list(discriminator.named_parameters())
                if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in list(generator.named_parameters())
                + list(discriminator.named_parameters())
                if any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )

    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        generator,
        discriminator,
        optimizer,
        train_loader,
        eval_loader,
        scheduler,
    ) = accelerator.prepare(
        generator,
        discriminator,
        optimizer,
        train_loader,
        eval_loader,
        scheduler,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        generator.train()
        discriminator.train()

        total_loss = 0.0
        total_gen_loss = 0.0
        total_disc_loss = 0.0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(generator, discriminator):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch.get("token_type_ids", None)
                generator_labels = batch["labels"]

                generator_outputs = generator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=generator_labels,
                )

                generator_loss = generator_outputs.loss

                with torch.no_grad():
                    corrupted_input_ids, discriminator_labels = make_electra_inputs(
                        input_ids=input_ids,
                        generator_logits=generator_outputs.logits,
                        generator_labels=generator_labels,
                    )

                discriminator_outputs = discriminator(
                    input_ids=corrupted_input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                disc_logits = discriminator_outputs.logits

                active_positions = attention_mask.view(-1) == 1
                disc_loss = F.binary_cross_entropy_with_logits(
                    disc_logits.view(-1)[active_positions],
                    discriminator_labels.float().view(-1)[active_positions],
                )

                loss = (
                    args.generator_loss_weight * generator_loss
                    + args.discriminator_loss_weight * disc_loss
                )

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.detach().float().item()
            total_gen_loss += generator_loss.detach().float().item()
            total_disc_loss += disc_loss.detach().float().item()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (
                completed_steps % args.logging_steps == 0
                and completed_steps > 0
                and accelerator.is_main_process
            ):
                avg_loss = total_loss / (step + 1)
                avg_gen = total_gen_loss / (step + 1)
                avg_disc = total_disc_loss / (step + 1)

                print(
                    f"epoch={epoch + 1} "
                    f"step={completed_steps} "
                    f"loss={avg_loss:.4f} "
                    f"generator_loss={avg_gen:.4f} "
                    f"discriminator_loss={avg_disc:.4f}"
                )

        generator.eval()
        discriminator.eval()

        eval_disc_loss = 0.0
        eval_steps = 0

        for batch in eval_loader:
            with torch.no_grad():
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch.get("token_type_ids", None)
                generator_labels = batch["labels"]

                generator_outputs = generator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=generator_labels,
                )

                corrupted_input_ids, discriminator_labels = make_electra_inputs(
                    input_ids=input_ids,
                    generator_logits=generator_outputs.logits,
                    generator_labels=generator_labels,
                )

                discriminator_outputs = discriminator(
                    input_ids=corrupted_input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                disc_logits = discriminator_outputs.logits
                active_positions = attention_mask.view(-1) == 1

                disc_loss = F.binary_cross_entropy_with_logits(
                    disc_logits.view(-1)[active_positions],
                    discriminator_labels.float().view(-1)[active_positions],
                )

                eval_disc_loss += disc_loss.detach().float().item()
                eval_steps += 1

        eval_disc_loss = eval_disc_loss / max(eval_steps, 1)

        if accelerator.is_main_process:
            print(
                f"epoch={epoch + 1} "
                f"eval_discriminator_loss={eval_disc_loss:.4f}"
            )

        if args.save_every_epoch:
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                epoch_dir = os.path.join(
                    args.output_dir,
                    f"checkpoint-epoch-{epoch + 1}",
                )
                os.makedirs(epoch_dir, exist_ok=True)

                unwrapped_disc = accelerator.unwrap_model(discriminator)
                unwrapped_gen = accelerator.unwrap_model(generator)

                unwrapped_disc.save_pretrained(
                    os.path.join(epoch_dir, "discriminator"),
                    safe_serialization=True,
                )
                unwrapped_gen.save_pretrained(
                    os.path.join(epoch_dir, "generator"),
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(
                    os.path.join(epoch_dir, "discriminator")
                )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = args.output_dir
        discriminator_dir = os.path.join(final_dir, "discriminator")
        generator_dir = os.path.join(final_dir, "generator")

        os.makedirs(discriminator_dir, exist_ok=True)
        os.makedirs(generator_dir, exist_ok=True)

        unwrapped_disc = accelerator.unwrap_model(discriminator)
        unwrapped_gen = accelerator.unwrap_model(generator)

        unwrapped_disc.save_pretrained(
            discriminator_dir,
            safe_serialization=True,
        )
        unwrapped_gen.save_pretrained(
            generator_dir,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(discriminator_dir)
        tokenizer.save_pretrained(generator_dir)

        print(f"Saved ELECTRA discriminator to: {discriminator_dir}")
        print(f"Saved ELECTRA generator to: {generator_dir}")
        print("For NLI fine-tuning, use the discriminator directory.")


if __name__ == "__main__":
    main()
