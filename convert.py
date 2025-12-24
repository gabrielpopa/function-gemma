import os
import argparse
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

# Setup the export configurations and parameters for text generation models.
export_config = ExportConfig()
export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
export_config.mask_as_input = True

# Convert to LiteRT-LM Format
converter.convert_to_litert(
    pytorch_model,
    output_path=litertlm_output_dir,
    output_name_prefix="mobile-actions",
    prefill_seq_len=256,
    kv_cache_max_len=1024,
    quantize="dynamic_int8",
    export_config=export_config,
    tokenizer_model_path=tokenizer_model_path,
    base_llm_metadata_path=metadata_path,
    output_format="litertlm",
    model_prompt_prefix="<start_of_turn>model\n",
    model_prompt_suffix="<end_of_turn>\n",
    user_prompt_prefix="<start_of_turn>user\n",
    user_prompt_suffix="<end_of_turn>\n",
)
