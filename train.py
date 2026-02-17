
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
    dataset_path: Optional[str]
    tools_path: Optional[str]
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
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
        default=["dataset.json"],
        help="Path to your dataset JSON files. Overrides --dataset_repo_id/--dataset_filename.",
    )
    parser.add_argument(
        "--tools_path",
        default=None,
        help="Path to your tools JSON file.",
    )
    parser.add_argument("--dataset_repo_id", default="google/mobile-actions")
    parser.add_argument("--dataset_filename", default="dataset.normalized.jsonl")
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

def main() -> None:
    args = _parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Pick one: --fp16 or --bf16.")
    set_seed(args.seed)
    if not args.local_files_only:
        _maybe_login(skip_login=args.skip_login)

    tools = None
    if args.tools_path:
        with open(args.tools_path, "r", encoding="utf-8") as f:
            tools = json.load(f)
    else:
        tools = [
            {
                "type": "function",
                "function": {
                "name": "search_web",
                "description": "searches the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "query": {"type": "string", "description": "query to be searched"}
                    },
                    "required": ["query"]
                }
                }
            }
        ]
    # tools = [json.dumps(tool, ensure_ascii=False) for tool in tools]

    DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"

    def create_conversation(sample):
        return {
            "messages": [
                {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
                {"role": "user", "content": sample["user_content"]},
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": sample["tool_name"], "arguments": json.loads(sample["tool_arguments"])} }]},
            ],
            "tools": tools
        }

    simple_tool_calling = None
    if args.tools_path:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            simple_tool_calling = json.load(f)
    else:
        simple_tool_calling = [
        {"user_content":"What is the reimbursement limit for travel meals?","tool_name":"search_web","tool_arguments":"{\"query\": \"travel meal reimbursement limit policy\"}"},
        {"user_content":"What is the current stock price of Google?","tool_name":"search_web","tool_arguments":"{\"query\": \"current Google stock price\"}"},
        {"user_content":"How do I configure the VPN for the New York office?","tool_name":"search_web","tool_arguments":"{\"query\": \"VPN configuration guide New York office\"}"},
        {"user_content":"Explain the difference between REST and GraphQL.","tool_name":"search_web","tool_arguments":"{\"query\": \"difference between REST and GraphQL\"}"},
        {"user_content":"Who is the product owner for Project Chimera?","tool_name":"search_web","tool_arguments":"{\"query\": \"Project Chimera product owner\"}"},
        {"user_content":"Find the documentation for the 'requests' library in Python.","tool_name":"search_web","tool_arguments":"{\"query\": \"Python requests library documentation\"}"},
        {"user_content":"What are the core values listed in our employee handbook?","tool_name":"search_web","tool_arguments":"{\"query\": \"employee handbook core values\"}"},
        {"user_content":"What is the weather forecast for the company retreat in Bali?","tool_name":"search_web","tool_arguments":"{\"query\": \"weather forecast Bali\"}"},
        {"user_content":"I need to reset my Okta password. How do I do that?","tool_name":"search_web","tool_arguments":"{\"query\": \"Okta password reset procedure\"}"},
        {"user_content":"Who won the World Series last year?","tool_name":"search_web","tool_arguments":"{\"query\": \"MLB World Series winner last year\"}"},
        {"user_content":"What is the guest Wi-Fi password for the 4th floor?","tool_name":"search_web","tool_arguments":"{\"query\": \"guest wifi password 4th floor\"}"},
    ]

    torch.cuda.empty_cache()
    
    dataset = Dataset.from_list(simple_tool_calling)

    # Convert dataset to conversational format
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

    # Split dataset into 80% training samples and 20% test samples
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
    # ------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        use_fast=False,          # IMPORTANT
    )
    print("args.local_files_only:", args.local_files_only)
    print("is_fast:", tokenizer.is_fast)
    print("len(tokenizer) =", len(tokenizer))

    print("vocab_files:", tokenizer.vocab_files_names)
    s = "<end_function_response>"
    print(tokenizer.encode(s, add_special_tokens=False))
    print("added vocab size =", tokenizer.added_tokens_encoder and len(tokenizer.added_tokens_encoder))
    print("special tokens =", tokenizer.special_tokens_map)
    tokenizer.save_pretrained(args.output_dir)
    # tokenizer.padding_side = "right"

    load_dtype = "auto"
    if args.fp16:
        load_dtype = torch.float16
    elif args.bf16:
        load_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=load_dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )

    print(f"Device: {model.device}")
    print(f"DType: {model.dtype}")

    # Print formatted user prompt
    print("--- dataset input ---")
    print(json.dumps(dataset["train"][0], indent=2))
    
    # if isinstance(tools, str):
    #     tools = json.loads(tools)
    # elif isinstance(tools, list) and tools and isinstance(tools[0], str):
    #     tools = [json.loads(t) for t in tools]
    debug_msg = tokenizer.apply_chat_template(dataset["train"][0]["messages"], tools=tools, add_generation_prompt=False, tokenize=False)
    print("--- Formatted prompt ---")
    print(debug_msg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    from trl import SFTConfig

    training_args = SFTConfig(
        output_dir=args.output_dir,                            # Directory to save adapters
        num_train_epochs=args.num_train_epochs,                               # Number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,                    # Batch size per device during training
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,                    # Gradient accumulation during training
        logging_strategy="steps",                         # Log every steps
        eval_strategy="steps",                            # Evaluate loss metrics based on steps
        eval_steps=50,                                    # Evaluate loss metrics every 50 steps
        logging_steps=10,                                 # Log loss metrics every 50 steps
        save_strategy="epoch",                            # Save checkpoint every epoch
        learning_rate=1e-5,                               # Learning rate,
        lr_scheduler_type="cosine",                       # Cosine scheduler is often better for full FT
        max_length=2048,                       # Max sequence length for model and packing of the dataset
        gradient_checkpointing=True,                      # Use gradient checkpointing to save memory
        packing=False,                                    # Groups multiple samples in the dataset into a single sequence
        optim="adamw_torch_fused",                        # Use fused adamw optimizer
        fp16=True if args.fp16 == torch.float16 else False,
        bf16=True if args.bf16 == torch.bfloat16 else False,                                        # Use bf16 for mixed precision training
        completion_only_loss=True,                        # Train on completion only to improve quality
        report_to="none"                                  # No reporting.
    )

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    tok = AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True, use_fast=False,)
    print("len(tok) =", len(tok))
    print("added vocab size =", tok.added_tokens_encoder and len(tok.added_tokens_encoder))
    print("special tokens =", tok.special_tokens_map)

    _maybe_write_chat_template_file(tokenizer, args.output_dir)
    _ensure_tokenizer_model_file(args.output_dir, args.model_id, local_files_only=args.local_files_only)

    print("Done. Saved to:", args.output_dir)

    # Access the log history
    log_history = trainer.state.log_history

    # Extract training / validation loss
    train_losses = [log["loss"] for log in log_history if "loss" in log]
    epoch_train = [log["epoch"] for log in log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]
    print("\nTraining losses:", list(zip(epoch_train, train_losses)))
    print("Evaluation losses:", list(zip(epoch_eval, eval_losses)))
    
    trained_model = AutoModelForCausalLM.from_pretrained(args.output_dir, device_map="auto")
    trained_tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=False,)

    from transformers import pipeline
    pipe = pipeline("text-generation", model=trained_model, tokenizer=trained_tokenizer)

    # Test a prompt
    user_prompt = "What is the time in Cairo and the weather in Rome?"  #@param {type:"string"}
    messages = [
        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": user_prompt}
    ]
    
    trained_prompt = trained_tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False)

    print(f"\n\033[1mPrompt:\033[0m {trained_prompt}")
    output = pipe(trained_prompt, max_new_tokens=512)

    print(f"\n\033[1mFine-tuned model output:\033[0m \n{output}")
    print(f"\n\033[1m--- End of Fine-tuned model ---\033[0m \n")

    print("\n\033[1mTraining losses:", list(zip(epoch_train, train_losses)), "\033[0m")
    print("\n\033[1mEvaluation losses:", list(zip(epoch_eval, eval_losses)), "\033[0m")
if __name__ == "__main__":
    main()
