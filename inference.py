#!/usr/bin/env python3
"""
Inference script for fine-tuned FunctionGemma model.
Tests the model's ability to generate function calls for mobile actions.
"""

import argparse
import os
import json
import torch
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, set_seed


def _parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned FunctionGemma model.")
    parser.add_argument("--model_path", default="functiongemma-270m-it-mobile-actions-sft")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tools_path", default="default_tools.json")
    parser.add_argument("--system_prompt_path", default="default_system_prompt.txt")
    parser.add_argument("--query", default=None, help="Single user query to test.")
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str, device: str):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model(model_path: str, device: str, dtype: torch.dtype):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"✓ Model loaded on {device} with dtype={dtype}")
    return model, tokenizer, processor


def _load_tools(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def create_prompt(user_request: str, tools: List[Dict[str, Any]], system_prompt: str):
    """Create a properly formatted prompt for the model."""
    return {
        "messages": [
            {
                "role": "developer",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_request
            }
        ],
        "tools": tools
    }


def test_model(model, tokenizer, processor, user_request, tools, system_prompt, max_new_tokens, temperature):
    """Test the model with a user request."""
    
    print(f"\n{'='*60}")
    print(f"User Request: {user_request}")
    print(f"{'='*60}")
    
    # Create prompt
    prompt_data = create_prompt(user_request, tools=tools, system_prompt=system_prompt)
    
    # Apply chat template
    prompt_text = processor.apply_chat_template(
        prompt_data["messages"],
        tools=prompt_data["tools"],
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"\nPrompt (first 500 chars):\n{prompt_text[:500]}...\n")
    
    # Generate
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        ).strip()

        print(f"Model Output:\n{generated_text}\n")

        # Try to parse function calls
        if "<start_function_call>" in generated_text:
            print("✓ Function call detected!")
            # Extract function calls
            import re
            calls = re.findall(r'<start_function_call>(.*?)<end_function_call>', generated_text, re.DOTALL)
            for call in calls:
                print(f"  - {call}")
        else:
            print("ℹ No function call generated")

        return generated_text

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run inference tests."""
    
    args = _parse_args()
    set_seed(args.seed)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    print("=" * 60)
    print("FunctionGemma Mobile Actions - Inference Test")
    print("=" * 60)
    
    # Load model
    model_path = args.model_path
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please ensure training has completed by running: python tra.py")
        return False
    
    model, tokenizer, processor = load_model(model_path, device=device, dtype=dtype)
    tools = _load_tools(args.tools_path)
    system_prompt = _load_system_prompt(args.system_prompt_path)
    
    # Test cases
    if args.query:
        test_requests = [args.query]
    else:
        test_requests = [
            "Turn on the flashlight",
            "Create a contact named John Smith with phone 555-1234",
            "Send an email to john@example.com with subject 'Meeting' and message 'See you tomorrow'",
            "Schedule a meeting called 'Q4 Review' for tomorrow at 2pm",
            "Show me the nearest coffee shop",
            "Open WiFi settings"
        ]
    
    print(f"\nRunning {len(test_requests)} inference tests...\n")
    
    results = []
    for i, request in enumerate(test_requests, 1):
        print(f"\n[Test {i}/{len(test_requests)}]")
        output = test_model(
            model,
            tokenizer,
            processor,
            request,
            tools,
            system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        results.append({
            "request": request,
            "output": output
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Tests completed: {len(results)}/{len(test_requests)}")
    
    # Save results
    results_file = "inference_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
