from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from countdown_distill.prompting import render_chat_prompt


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for f in features:
            curr_len = len(f["input_ids"])
            pad_len = max_len - curr_len

            padded_input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            padded_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "labels": torch.tensor(padded_labels),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA-SFT for google/gemma-3-1b-it on Countdown.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--base-model-id", default="google/gemma-3-1b-it", help="Base model when --model-id points to a LoRA adapter.")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-val-examples", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-module", action="append", default=[])
    return parser.parse_args()


def build_quantization_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    if platform.system() == "Windows":
        raise RuntimeError("bitsandbytes 4-bit training is usually unreliable on native Windows. Use WSL2/Linux or drop --load-in-4bit.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )


def build_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def tokenize_example(example: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, Any]:
    messages = example["messages"]
    prompt_messages = messages[:-1]

    full_text = render_chat_prompt(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_chat_prompt(tokenizer, prompt_messages, add_generation_prompt=True)

    tokenized_full = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding=False,)
    tokenized_prompt = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, padding=False,)

    labels = list(tokenized_full["input_ids"]).copy()
    prompt_length = len(tokenized_prompt["input_ids"])
    for index in range(min(prompt_length, len(labels))):
        labels[index] = -100

    if len(labels) > max_seq_length:
        labels = labels[:max_seq_length]
        tokenized_full["input_ids"] = tokenized_full["input_ids"][:max_seq_length]
        tokenized_full["attention_mask"] = tokenized_full["attention_mask"][:max_seq_length]

    tokenized_full["labels"] = labels
    return tokenized_full


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_jsonl),
            "validation": str(args.val_jsonl),
        },
    )
    if args.max_train_examples > 0:
        train_limit = min(args.max_train_examples, len(dataset["train"]))
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(train_limit))
    if args.max_val_examples > 0:
        val_limit = min(args.max_val_examples, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(val_limit))

    model_path = Path(args.model_id)
    adapter_config_path = model_path / "adapter_config.json"
    tokenizer_source = model_path if adapter_config_path.exists() and (model_path / "tokenizer_config.json").exists() else args.model_id
    if adapter_config_path.exists() and not (model_path / "tokenizer_config.json").exists():
        tokenizer_source = args.base_model_id

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = build_quantization_config(args.load_in_4bit)
    model_kwargs: dict[str, Any] = {"torch_dtype": build_dtype()}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    if adapter_config_path.exists():
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id, **model_kwargs)
        if args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            base_model.config.use_cache = False
        if quantization_config is not None:
            base_model = prepare_model_for_kbit_training(base_model)
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_module or DEFAULT_LORA_TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)

    tokenized = dataset.map(
        lambda batch: tokenize_example(batch, tokenizer, args.max_seq_length),
        remove_columns=dataset["train"].column_names,
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=SupervisedDataCollator(tokenizer),
    )

    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    run_config = {
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "model_id": args.model_id,
        "base_model_id": args.base_model_id,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "lora_target_modules": args.lora_target_module or DEFAULT_LORA_TARGET_MODULES,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
