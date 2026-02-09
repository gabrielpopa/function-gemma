import os
import argparse
import torch
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig
from ai_edge_torch.generative.layers import kv_cache

# Metadata for FunctionGemma
llm_metadata = r"""start_token: {
    token_ids: {
        ids: [ 2 ]
    }
}
stop_tokens: {
    token_str: "<end_of_turn>"
}
stop_tokens: {
    token_str: "<eos>"
}
llm_model_type: {
    function_gemma: {}
}
"""

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a fine-tuned FunctionGemma checkpoint to LiteRT-LM format.")
    parser.add_argument("--checkpoint_dir", default="functiongemma-270m-it-mobile-actions-sft")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument(
        "--output_name_prefix",
        default="mobile-actions",
        help="Prefix for the exported LiteRT-LM model files.",
    )
    parser.add_argument(
        "--prefill_seq_len",
        type=int,
        default=256,
        help="Sequence length for the prefill signature (must be > 0).",
    )
    parser.add_argument(
        "--kv_cache_max_len",
        type=int,
        default=1024,
        help="Max KV cache length for decode (must be > 0 and >= prefill_seq_len).",
    )
    parser.add_argument(
        "--quantize",
        default="dynamic_int8",
        choices=[
            "none",
            "dynamic_int8",
            "weight_only_int8",
            "fp16",
            "dynamic_int4_block32",
            "dynamic_int4_block128",
        ],
        help=(
            "Quantization mode. Use 'none' to disable. "
            "dynamic_int8 is a good default for size/speed."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the PyTorch export step (cpu, cuda, or cuda:0). Defaults to cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype to cast the PyTorch model before export. Defaults to float32.",
    )
    return parser.parse_args()


args = _parse_args()

checkpoint_dir = args.checkpoint_dir
litertlm_output_dir = args.output_dir
os.makedirs(litertlm_output_dir, exist_ok=True)

tokenizer_model_path = os.path.join(checkpoint_dir, "tokenizer.model")
if not os.path.exists(tokenizer_model_path):
    raise FileNotFoundError(
        f"Missing `{tokenizer_model_path}`.\n"
        "Fix: re-run `train.py` (it now exports tokenizer.model) or copy it from the base model repo into the "
        "checkpoint directory."
    )

# Create the LLM metadata file
metadata_path = os.path.join(litertlm_output_dir, 'base_llm_metadata.textproto')
with open(metadata_path, 'w') as f:
  f.write(llm_metadata)

# Import the weights and build the PyTorch model
pytorch_model = gemma3.build_model_270m(checkpoint_dir)

print(f"--- Requested device: {args.device}")
requested_device = args.device
if requested_device != "cpu":
    if not torch.cuda.is_available():
        raise RuntimeError("Requested a CUDA device, but torch.cuda.is_available() is False.")
dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
model_dtype = dtype_map[args.dtype]
if requested_device == "cpu" and model_dtype != torch.float32:
    raise RuntimeError("CPU export only supports float32. Use --dtype float32 or set --device cuda.")
if args.prefill_seq_len <= 0:
    raise ValueError("--prefill_seq_len must be > 0.")
if args.kv_cache_max_len <= 0:
    raise ValueError("--kv_cache_max_len must be > 0.")
if args.prefill_seq_len > args.kv_cache_max_len:
    raise ValueError("--kv_cache_max_len must be >= --prefill_seq_len.")
pytorch_model = pytorch_model.to(device=requested_device, dtype=model_dtype)
pytorch_model.eval()

# Setup the export configurations and parameters for text generation models.
export_config = ExportConfig()
export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
export_config.mask_as_input = True

# Convert to LiteRT-LM Format
try:
    converter.convert_to_litert(
        pytorch_model,
        output_path=litertlm_output_dir,
        output_name_prefix=args.output_name_prefix,
        prefill_seq_len=args.prefill_seq_len,
        kv_cache_max_len=args.kv_cache_max_len,
        quantize=args.quantize,
        export_config=export_config,
        tokenizer_model_path=tokenizer_model_path,
        base_llm_metadata_path=metadata_path,
        output_format="litertlm",
        model_prompt_prefix="<start_of_turn>model\n",
        model_prompt_suffix="<end_of_turn>\n",
        user_prompt_prefix="<start_of_turn>user\n",
        user_prompt_suffix="<end_of_turn>\n",
    )
except RuntimeError as exc:
    print(f"Error during conversion: {exc}")
    if requested_device == "cpu":
        raise
    print(
        "GPU export failed (likely torch.export fake tensor device mismatch). "
        "Falling back to CPU export with float32.\n"
        f"Reason: {exc}"
    )
    pytorch_model = pytorch_model.to(device="cpu", dtype=torch.float32)
    converter.convert_to_litert(
        pytorch_model,
        output_path=litertlm_output_dir,
        output_name_prefix=args.output_name_prefix,
        prefill_seq_len=args.prefill_seq_len,
        kv_cache_max_len=args.kv_cache_max_len,
        quantize=args.quantize,
        export_config=export_config,
        tokenizer_model_path=tokenizer_model_path,
        base_llm_metadata_path=metadata_path,
        output_format="litertlm",
        model_prompt_prefix="<start_of_turn>model\n",
        model_prompt_suffix="<end_of_turn>\n",
        user_prompt_prefix="<start_of_turn>user\n",
        user_prompt_suffix="<end_of_turn>\n",
    )
