# Training Notes (train.py)

This document summarizes how `train.py` works, key settings, how to read logs, and what to change when training looks wrong.

## What train.py does
1. Loads a dataset from JSON/JSONL (via `load_dataset("json", ...)`).
2. Formats each example into `prompt` and `completion` using the model’s chat template.
3. Splits into `train` and `eval` by the `metadata` field.
4. Fine‑tunes the model with supervised fine‑tuning (SFT).
5. Saves the fine‑tuned checkpoint and tokenizer.

## Required dataset fields
Each record must include:
1. `messages` (list)
2. `tools` (list)
3. `metadata` (`train` or `eval`)

## Key parameters

- `--dataset_path`  
  Path to your JSON/JSONL dataset. If not provided, it pulls from Hugging Face.

- `--output_dir`  
  Where the fine‑tuned model is saved.

- `--num_train_epochs`  
  Number of full passes over the dataset. Increase for small datasets.

- `--per_device_train_batch_size`  
  Batch size per device. Lower if you run out of memory.

- `--gradient_accumulation_steps`  
  Accumulates gradients over multiple steps to simulate a larger batch size.  
  Effective batch size = `per_device_train_batch_size * gradient_accumulation_steps`.

- `--learning_rate`  
  Learning rate for training. Smaller datasets often need smaller LR to avoid overfitting.

- `--max_length`  
  Max token length for prompt + completion. Must be large enough for full tool calls.

- `--eval_steps`, `--eval_strategy`  
  How often eval runs (default is steps).  
  If you have no eval data, `eval_loss` will be `nan`.

- `--bf16` / `--fp16`  
  Mixed precision options (CUDA only). CPU uses float32.

## How to interpret logs

Healthy signals:
1. **Loss > 0** and usually decreases.
2. **grad_norm > 0**, meaning gradients are flowing.
3. **eval_loss** is a real number (not `nan`).

Red flags:
1. `loss=0.0` and `grad_norm=0.0`: labels are fully masked or completions are empty.
2. `eval_loss=nan`: eval split is missing or empty.
3. Loss decreases too fast and eval loss rises: overfitting.

## Common fixes when training looks wrong

If loss is **0.0**:
1. Ensure the **assistant tool call is the last message** in each example.
2. Make sure `completion` is non‑empty in the formatted dataset output.
3. Increase `--max_length` to avoid truncating completions.

If eval loss is **nan**:
1. Add `eval` records (even a small number helps).
2. Ensure `metadata` is exactly `eval` (case‑sensitive).

If training finishes too quickly:
1. Increase `--num_train_epochs` (e.g., 5–20).
2. Lower `--gradient_accumulation_steps` to get more updates per epoch.

If the model **overfits**:
1. Reduce `--num_train_epochs`.
2. Reduce `--learning_rate`.
3. Increase eval split size.

## Suggested starting values for small datasets

1. `--num_train_epochs 5` to `20`
2. `--learning_rate 1e-5` or `5e-6`
3. `--gradient_accumulation_steps 1` to `4`
4. `--max_length 1024` to `4096` (depending on prompt length)

## Minimal example

```bash
python train.py \
  --dataset_path dataset.jsonl \
  --output_dir DATASET \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_length 4096 \
  --bf16
```
