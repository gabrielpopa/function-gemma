
import argparse
import json
import os
import shutil
from dataclasses import dataclass
from random import randint
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass(frozen=True)
class ScriptArgs:
    model_id: str
    output_dir: str
    seed: int
    max_length: Optional[int]
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    attn_implementation: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool
    skip_login: bool
    local_files_only: bool
    run_smoke_tests: bool
    dataset_path: Optional[List[str]]
    dataset_repo_id: str
    dataset_filename: str
    download_example_dataset: Optional[str]


@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            if len(label) < max_len:
                padded = label + [self.label_pad_token_id] * (max_len - len(label))
            else:
                padded = label[:max_len]
            padded_labels.append(padded)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def _parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma 270M on the Mobile Actions dataset.")
    parser.add_argument("--model_id", default="google/functiongemma-270m-it")
    parser.add_argument("--output_dir", default="functiongemma-270m-it-mobile-actions-sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max sequence length. If omitted, computes it from the longest training example (+100).",
    )
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_login", action="store_true", help="Skip Hugging Face login even if HF_TOKEN is set.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Disable network access; use only local Hugging Face cache.",
    )
    parser.add_argument(
        "--dataset_path",
        nargs="+",
        default=None,
        help="Path(s) to your dataset JSONL files. Overrides --dataset_repo_id/--dataset_filename.",
    )
    parser.add_argument("--dataset_repo_id", default="google/mobile-actions")
    parser.add_argument("--dataset_filename", default="dataset.jsonl")
    parser.add_argument(
        "--download_example_dataset",
        default=None,
        help="If set, downloads the example dataset to this path and exits.",
    )
    parser.add_argument(
        "--run_smoke_tests",
        action="store_true",
        help="Runs a small generation before/after training (slow; requires GPU for practicality).",
    )

    ns = parser.parse_args()
    return ScriptArgs(**vars(ns))


def _maybe_login(skip_login: bool) -> None:
    if skip_login:
        return
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable not set. "
            "Set it to access gated models/datasets (or pass --skip_login if you have local cache access)."
        )
    login(token=hf_token)


def _download_example_dataset(
    output_path: str, repo_id: str, filename: str, local_files_only: bool
) -> str:
    data_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_files_only=local_files_only,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    shutil.copyfile(data_file, output_path)
    return output_path


def _load_dataset(args: ScriptArgs) -> Dataset:
    if args.dataset_path:
        data_files = args.dataset_path
        return load_dataset("json", data_files=data_files, split="train")

    data_file = hf_hub_download(
        repo_id=args.dataset_repo_id,
        filename=args.dataset_filename,
        repo_type="dataset",
        local_files_only=args.local_files_only,
    )
    return load_dataset("json", data_files=data_file, split="train")


def _format_prompt_completion(
    tokenizer: Any, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
) -> Dict[str, str]:
    prompt_and_completion = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = prompt_and_completion[len(prompt) :]
    return {"prompt": prompt, "completion": completion}


def _ensure_tokenizer_model_file(output_dir: str, base_model_id: str, local_files_only: bool) -> None:
    out_path = os.path.join(output_dir, "tokenizer.model")
    if os.path.exists(out_path):
        return

    src_path = hf_hub_download(repo_id=base_model_id, filename="tokenizer.model", local_files_only=local_files_only)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src_path, out_path)


def _maybe_write_chat_template_file(tokenizer: Any, output_dir: str) -> None:
    path = os.path.join(output_dir, "chat_template.jinja")
    if os.path.exists(path):
        return
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(chat_template)


def _smoke_test_generation(model_id: str, tokenizer: Any, tools: List[Dict[str, Any]]) -> None:
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model_id, tokenizer=tokenizer)
    user_prompt = 'Schedule a "team meeting" tomorrow at 4pm.'
    messages = [
        {
            "role": "developer",
            "content": (
                "Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2024-11-15T05:59:00.\n"
                "Day of week is Friday\n"
                "You are a model that can do function calling with the following functions\n"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    out = pipe(prompt, max_new_tokens=256, do_sample=False)
    print("\nPrompt:\n", user_prompt)
    print("\nModel output:\n", out[0]["generated_text"][len(prompt) :].strip())


def main() -> None:
    args = _parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Pick one: --fp16 or --bf16.")
    set_seed(args.seed)
    if not args.local_files_only:
        _maybe_login(skip_login=args.skip_login)

    torch.cuda.empty_cache()

    if args.download_example_dataset:
        downloaded_path = _download_example_dataset(
            output_path=args.download_example_dataset,
            repo_id=args.dataset_repo_id,
            filename=args.dataset_filename,
            local_files_only=args.local_files_only,
        )
        print("Downloaded dataset to:", downloaded_path)
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, local_files_only=args.local_files_only)
    except Exception as e:
        print(
            "Warning: failed to load slow tokenizer (use_fast=False). "
            "Falling back to fast tokenizer; will still export tokenizer.model separately.\n"
            f"Reason: {e}"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, local_files_only=args.local_files_only)
    tokenizer.padding_side = "right"

    raw_dataset = _load_dataset(args).shuffle(seed=args.seed)
    print(
        "\nHere's an example from the dataset:\n",
        json.dumps(raw_dataset[randint(0, len(raw_dataset) - 1)], indent=2),
    )

    def apply_format(example: Dict[str, Any]) -> Dict[str, Any]:
        pc = _format_prompt_completion(tokenizer, example["messages"], example["tools"])
        return {"prompt": pc["prompt"], "completion": pc["completion"], "split": example["metadata"]}

    processed = raw_dataset.map(apply_format, remove_columns=raw_dataset.column_names)

    print("\nHere's an example from the formatted dataset:\n", json.dumps(processed[randint(0, len(processed) - 1)], indent=2))

    if args.max_length is None:
        longest = max(processed, key=lambda e: len(e["prompt"] + e["completion"]))
        longest_tokens = len(tokenizer.tokenize(longest["prompt"] + longest["completion"]))
        max_length = longest_tokens + 100
        print(f"\nLongest example has {longest_tokens} tokens; using max_length={max_length}.")
    else:
        max_length = args.max_length
        print(f"\nUsing provided max_length={max_length}.")

    train_dataset = processed.filter(lambda e: e["split"] == "train")
    eval_dataset = processed.filter(lambda e: e["split"] == "eval")

    if args.run_smoke_tests:
        tools_for_test = raw_dataset[0]["tools"]
        print("\nBase model smoke test:")
        _smoke_test_generation(args.model_id, tokenizer, tools_for_test)

    if args.fp16:
        model_dtype = torch.float16
    elif args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = None
    model_kwargs = dict(
        device_map="auto",
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=model_dtype,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=model_dtype,
            **model_kwargs,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tokenize_with_labels(example: Dict[str, str]) -> Dict[str, Any]:
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        full = tokenizer(
            example["prompt"] + example["completion"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        labels = list(full["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        full["labels"] = labels
        return full

    train_tok = train_dataset.map(tokenize_with_labels, remove_columns=train_dataset.column_names)
    eval_tok = eval_dataset.map(tokenize_with_labels, remove_columns=eval_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _maybe_write_chat_template_file(tokenizer, args.output_dir)
    _ensure_tokenizer_model_file(args.output_dir, args.model_id, local_files_only=args.local_files_only)

    print("Done. Saved to:", args.output_dir)

    if args.run_smoke_tests:
        print("\nFine-tuned model smoke test:")
        _smoke_test_generation(args.output_dir, tokenizer, raw_dataset[0]["tools"])


if __name__ == "__main__":
    main()
