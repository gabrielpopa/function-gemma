
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

import json
from random import randint
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

data_file = hf_hub_download(repo_id="google/mobile-actions", filename="dataset.jsonl", repo_type="dataset")
dataset = load_dataset("text", data_files=data_file, encoding="utf-8")["train"].shuffle()

print(f"\n\033[1mHere's an example from your dataset:\033[0m \n{json.dumps(json.loads(dataset[randint(0, len(dataset) - 1)]['text']), indent=2)}")



import json

def apply_format(sample):
  template_iputs = json.loads(sample['text'])

  prompt_and_completion = tokenizer.apply_chat_template(
    template_iputs['messages'],
    tools=template_iputs['tools'],
    tokenize=False,
    # add_generation_prompt is False since we don't need model output after all
    # messages.
    add_generation_prompt=False)

  prompt = tokenizer.apply_chat_template(
    template_iputs['messages'][:-1],
    tools=template_iputs['tools'],
    tokenize=False,
    # add_generation_prompt is True since we would like to include
    # "model" in the prompt, if needed.
    add_generation_prompt=True)

  completion = prompt_and_completion[len(prompt):]

  return {
     "prompt": prompt,
     "completion": completion,
     "split": template_iputs["metadata"],
  }

processed_dataset = dataset.map(apply_format)
     
#in [9]

#@title Review the processed dataset

print("\033[1mHere's an example from the formatted dataset:\033[0m")
print(json.dumps(processed_dataset[randint(0, len(processed_dataset) - 1)], indent=2))

longest_example = max(processed_dataset, key=lambda example: len(example['prompt'] + example['completion']))
longest_example_token_count = len(tokenizer.tokenize(longest_example['prompt'] + longest_example['completion']))

print(f"\n\033[1mThe longest example length is {len(longest_example['prompt'] + longest_example['completion'])} with {longest_example_token_count} tokens. We need to set the max_length larger than the token count in SFTConfig below.\033[0m")
print(json.dumps(longest_example, indent=2))

max_token_count = longest_example_token_count + 100
print(f"\n\033[1mUsing max_token_count of {max_token_count} (= {longest_example_token_count} + 100) for training below.\033[0m")
     
# In [10]
# ------- @title Prepare train and eval dataset.

train_dataset = processed_dataset.filter(lambda example: example['split'] == 'train')
eval_dataset = processed_dataset.filter(lambda example: example['split'] == 'eval')

import json

# ------- @title Test with a prompt

from transformers import pipeline
from random import randint
import re

# Create a transformers inference pipeline
pipe = pipeline("text-generation", model=gemma_model, tokenizer=tokenizer)

user_prompt = "Schedule a \"team meeting\" tomorrow at 4pm."  #@param {type:"string"}
messages = [
    {"role": "developer", "content": "Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2024-11-15T05:59:00. You are a model that can do function calling with the following functions"},
    {"role": "user", "content": user_prompt}
]

# Reuse the tools from the sample
tools = json.loads(dataset[0]['text'])['tools']

prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True)

print(f"\n\033[1mPrompt:\033[0m {user_prompt}")
output = pipe(prompt, max_new_tokens=max_token_count)
model_output = output[0]['generated_text'][len(prompt):].strip()

print(f"\n\033[1mBase model output:\033[0m {model_output}")

# ---- Test with training dataset

from transformers import pipeline
from random import randint
import re

# Create a transformers inference pipeline
pipe = pipeline("text-generation", model=gemma_model, tokenizer=tokenizer)

# Select a random sample from the test dataset
rand_idx = randint(0, len(train_dataset) - 1)
test_sample = train_dataset[rand_idx]

input_prompt = test_sample['prompt']
expected_output = test_sample['completion']

# Generate the output
output = pipe(input_prompt, max_new_tokens=max_token_count, skip_special_tokens=False)
actual_output = output[0]['generated_text'][len(input_prompt):].strip()

print(f"\n\033[1mInput prompt\033[0m   : {input_prompt}")
print(f"\n\033[1mExpected output\033[0m: {expected_output}")
print(f"\n\033[1mActual output\033[0m  : {actual_output}")

# --------- Fine-tune the model

import torch
from transformers import AutoModelForCausalLM
from trl import SFTConfig

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


# Test the fine-tuned model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Create Transformers inference pipeline
trained_model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(output_dir)
pipe = pipeline("text-generation", model=trained_model, tokenizer=tokenizer)
pipe_base = pipeline("text-generation", model=gemma_model, device_map="auto")

# Test a prompt
user_prompt = "Schedule a \"team meeting\" tomorrow at 4pm."  #@param {type:"string"}
messages = [
    {"role": "developer", "content": "Current date and time given in YYYY-MM-DDTHH:MM:SS format: 2024-11-15T05:59:00. You are a model that can do function calling with the following functions"},
    {"role": "user", "content": user_prompt}
]

# Reuse the tools from the sample
tools = json.loads(dataset[0]['text'])['tools']

prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True)

print(f"\n\033[1mPrompt:\033[0m {prompt}")
output = pipe(prompt, max_new_tokens=max_token_count)
output_base = pipe_base(prompt, max_new_tokens=max_token_count)
model_output = output[0]['generated_text'][len(prompt):].strip()
model_output_base = output_base[0]['generated_text'][len(prompt):].strip()

print(f"\n\033[1mFine-tuned model output:\033[0m {model_output}")

print(f"\n\033[1mBase model output:\033[0m       {model_output_base}")
     

