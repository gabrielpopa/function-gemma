
import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise RuntimeError("HF_TOKEN environment variable not set")

login(hf_token)
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# Clear GPU cache
torch.cuda.empty_cache()

gemma_model = "google/functiongemma-270m-it"

base_model = AutoModelForCausalLM.from_pretrained(
    gemma_model,
    device_map="auto",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(gemma_model)

print(f"Device: {base_model.device}")
print(f"DType:  {base_model.dtype}")

## FunctionGemma uses an AutoProcessor (not just tokenizer) for tool + chat templating
processor = AutoProcessor.from_pretrained(gemma_model)

print(f"Processor Device: {getattr(processor, 'device', 'N/A')}")

#----------------------------
# 2) Load Mobile Actions dataset
# ----------------------------
#


from datasets import load_dataset
ds = load_dataset("google/mobile-actions")  # default split: train (9.65k rows)

# Create a small validation split (the dataset is train-only)
ds = ds["train"].train_test_split(test_size=0.02, seed=42)
train_ds, eval_ds = ds["train"], ds["test"]

# ----------------------------
# 3) Format each example into FunctionGemma training text
#    Dataset schema includes:
#      - tools: list of tool schemas
#      - messages: list of chat messages with tool_calls on assistant turn
# ----------------------------
def to_text(example):
    text = processor.apply_chat_template(
        example["messages"],
        tools=example["tools"],
        tokenize=False,
        add_generation_prompt=False,  # IMPORTANT: for SFT include the assistant answer in the text
    )
    return {"text": text}

train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(to_text,  remove_columns=eval_ds.column_names)

# ----------------------------
# 4) Training config
# ----------------------------
output_dir = "functiongemma-270m-it-mobile-actions-sft"

from trl import SFTConfig

args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,   # Reduced from 4
    per_device_eval_batch_size=2,    # Reduced from 4
    gradient_accumulation_steps=8,   # Increased to 16 (2*8)
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=torch.cuda.is_available(),  # you already got bf16
    report_to="none",
    gradient_checkpointing=True,     # Save memory with gradient checkpointing
)

trainer = SFTTrainer(
    model=base_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=processor,      # TRL uses this to tokenize
)

# ----------------------------
# 5) Train + save
# ----------------------------
trainer.train()
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

print("Done. Saved to:", output_dir)
