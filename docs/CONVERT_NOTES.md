# convert.py Notes

This document explains the `convert.py` parameters, when to change them, and recommended defaults.

## Purpose
`convert.py` exports a fine‑tuned FunctionGemma checkpoint to LiteRT‑LM format using `ai_edge_torch`. It builds the model, sets export configuration, and writes a LiteRT‑LM bundle to `--output_dir`.

## Parameters

- `--checkpoint_dir`  
  Path to the fine‑tuned checkpoint directory (must contain `tokenizer.model`).  
  Default: `functiongemma-270m-it-mobile-actions-sft`

- `--output_dir`  
  Destination directory for LiteRT‑LM artifacts.  
  Default: `output`

- `--device`  
  Device for export: `cpu`, `cuda`, or `cuda:0`.  
  Default: `cpu`  
  Note: CPU export only supports `--dtype float32`.

- `--dtype`  
  Model dtype before export: `float32`, `float16`, `bfloat16`.  
  Default: `float32`  
  Use `float16/bfloat16` only with CUDA.

- `--output_name_prefix`  
  File prefix for exported artifacts.  
  Default: `mobile-actions`

- `--prefill_seq_len`  
  Sequence length used for the **prefill** signature.  
  Default: `256`  
  Increase if you expect longer prompts at inference time.

- `--kv_cache_max_len`  
  Max KV cache length for **decode**. Must be `>= prefill_seq_len`.  
  Default: `1024`  
  Increase if you expect long generations.

- `--quantize`  
  Quantization mode.  
  Default: `dynamic_int8`
  Allowed values:
  - `none` (no quantization, largest model, best fidelity)
  - `dynamic_int8` (good size/speed tradeoff)
  - `weight_only_int8` (weights only, faster than full precision)
  - `fp16` (smaller than fp32, requires CUDA)
  - `dynamic_int4_block32` (smaller, higher quality than block128, slower)
  - `dynamic_int4_block128` (smaller, faster, lower quality)

## Recommended Defaults (General)

- `--prefill_seq_len 256`  
  Works well for short tool‑calling prompts.

- `--kv_cache_max_len 1024`  
  Enough for moderate generations. Increase if you need longer outputs.

- `--quantize dynamic_int8`  
  Best balance of size and quality for most use cases.

If you need maximum quality and size doesn’t matter, use `--quantize none`.  
If you need minimum size and can tolerate quality loss, try `dynamic_int4_block128`.

## Example Usage

```bash
python convert.py \
  --checkpoint_dir functiongemma-270m-it-mobile-actions-sft \
  --output_dir output \
  --output_name_prefix mobile-actions \
  --prefill_seq_len 256 \
  --kv_cache_max_len 1024 \
  --quantize dynamic_int8
```
