#!/usr/bin/env python3
"""
Inference script for fine-tuned FunctionGemma model.
Tests the model's ability to generate function calls.
"""

import argparse
import os
import json
import re
import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, set_seed
from tqdm import tqdm


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
    parser.add_argument("--dataset_path", default="dataset.jsonl")
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
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


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of records.")
    return data


def _extract_expected_call(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls = msg.get("tool_calls")
            if isinstance(calls, list) and calls:
                fn = calls[0].get("function", {})
                return {
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments", {}),
                }
    return None


def _strip_to_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trimmed = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            break
        trimmed.append(msg)
    return trimmed


def _extract_tool_call_from_text(text: str) -> Optional[Dict[str, Any]]:
    if "<start_function_call>" in text:
        matches = re.findall(
            r"<start_function_call>(.*?)<end_function_call>",
            text,
            flags=re.DOTALL,
        )
        for match in matches:
            payload = match.strip()
            # Format: call:tool_name{key:value,...}
            if payload.startswith("call:") and "{" in payload and payload.endswith("}"):
                try:
                    name_part, args_part = payload.split("{", 1)
                    tool_name = name_part.replace("call:", "").strip()
                    args_text = args_part[:-1] if args_part.endswith("}") else args_part
                    args_text = args_text.replace("<escape>", "")
                    args_obj: Dict[str, Any] = {}
                    if args_text.strip():
                        parts = [p for p in args_text.split(",") if p.strip()]
                        for part in parts:
                            if ":" not in part:
                                continue
                            key, value = part.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            if value.lower() in {"none", "null"}:
                                parsed_val: Any = None
                            elif value.lower() == "true":
                                parsed_val = True
                            elif value.lower() == "false":
                                parsed_val = False
                            else:
                                if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
                                    value = value[1:-1]
                                parsed_val = value
                            if key:
                                args_obj[key] = parsed_val
                    return {"name": tool_name, "arguments": args_obj}
                except Exception:
                    pass
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    if "function" in parsed:
                        fn = parsed["function"]
                        return {"name": fn.get("name"), "arguments": fn.get("arguments", {})}
                    if "name" in parsed:
                        return {"name": parsed.get("name"), "arguments": parsed.get("arguments", {})}
            except json.JSONDecodeError:
                continue
    # Fallback: try to parse a JSON object from the whole text
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        snippet = text[left : right + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                if "function" in parsed:
                    fn = parsed["function"]
                    return {"name": fn.get("name"), "arguments": fn.get("arguments", {})}
                if "name" in parsed:
                    return {"name": parsed.get("name"), "arguments": parsed.get("arguments", {})}
        except json.JSONDecodeError:
            pass
    return None


def _compare_args(expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
    """Compare arguments with fuzzy matching for strings.
    
    For string values, accepts if one is a substring of the other (e.g., handles trailing punctuation).
    For other types, requires exact match.
    """
    if expected == predicted:
        return True
    
    # Check if keys match
    if set(expected.keys()) != set(predicted.keys()):
        return False
    
    # Compare each argument with fuzzy matching for strings
    for key in expected.keys():
        exp_val = expected[key]
        pred_val = predicted[key]
        
        # Exact match
        if exp_val == pred_val:
            continue
        
        # For strings, check if one contains the other (handles punctuation differences)
        if isinstance(exp_val, str) and isinstance(pred_val, str):
            exp_stripped = exp_val.strip()
            pred_stripped = pred_val.strip()
            if exp_stripped in pred_stripped or pred_stripped in exp_stripped:
                continue
        
        # No match found
        return False
    
    return True


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
    print("FunctionGemma - Inference Test")
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
    
    if args.query:
        print("\nSingle-query mode.\n")
        _ = test_model(
            model,
            tokenizer,
            processor,
            args.query,
            tools,
            system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        return True

    dataset = _load_dataset(args.dataset_path)
    eval_records = [r for r in dataset if r.get("metadata") == args.split]
    if not eval_records:
        print(f"No records found for split '{args.split}'.")
        return False

    results = []
    tool_stats: Dict[str, Dict[str, int]] = {}

    for i, record in tqdm(enumerate(eval_records, 1), total=len(eval_records), desc="Running", unit="test"):
        messages = record.get("messages", [])
        tools_for_record = record.get("tools", tools)
        expected = _extract_expected_call(messages)
        if expected is None:
            results.append({"index": i, "status": "no_expected_call"})
            continue

        prompt_messages = _strip_to_prompt(messages)
        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tools=tools_for_record,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        do_sample = args.temperature > 0
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
            )
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        ).strip()

        predicted = _extract_tool_call_from_text(generated_text)
        expected_name = expected.get("name")
        expected_args = expected.get("arguments", {})
        pred_name = predicted.get("name") if predicted else None
        pred_args = predicted.get("arguments", {}) if predicted else None

        tool_key = expected_name or "(unknown)"
        stats = tool_stats.setdefault(
            tool_key,
            {"total": 0, "pass": 0, "wrong_tool": 0, "wrong_params": 0, "no_call": 0},
        )
        stats["total"] += 1

        status = "pass"
        if predicted is None or pred_name is None:
            status = "no_call"
            stats["no_call"] += 1
        elif pred_name != expected_name:
            status = "wrong_tool"
            stats["wrong_tool"] += 1
        elif not _compare_args(expected_args, pred_args or {}):
            status = "wrong_params"
            stats["wrong_params"] += 1
        else:
            stats["pass"] += 1

        results.append(
            {
                "index": i,
                "user_message": next(
                    (m.get("content") for m in messages if m.get("role") == "user"),
                    None,
                ),
                "model_output": generated_text,
                "expected_tool": expected_name,
                "expected_args": expected_args,
                "pred_tool": pred_name,
                "pred_args": pred_args,
                "status": status,
            }
        )

    passed = sum(1 for r in results if r.get("status") == "pass")
    wrong_tool_total = sum(1 for r in results if r.get("status") == "wrong_tool")
    wrong_params_total = sum(1 for r in results if r.get("status") == "wrong_params")
    no_call_total = sum(1 for r in results if r.get("status") == "no_call")
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(
        "Total: {total} | Passed: {passed} | Failed: {failed} | "
        "Wrong Tool: {wrong_tool} | Wrong Params: {wrong_params} | No Call: {no_call}".format(
            total=len(results),
            passed=passed,
            failed=failed,
            wrong_tool=wrong_tool_total,
            wrong_params=wrong_params_total,
            no_call=no_call_total,
        )
    )

    headers = ["Tool", "Total", "Pass", "Wrong Tool", "Wrong Params", "No Call"]
    rows = []
    for tool_name, stats in sorted(tool_stats.items()):
        rows.append(
            [
                tool_name,
                str(stats["total"]),
                str(stats["pass"]),
                str(stats["wrong_tool"]),
                str(stats["wrong_params"]),
                str(stats["no_call"]),
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(values: List[str]) -> str:
        return " | ".join(values[i].ljust(col_widths[i]) for i in range(len(values)))

    print("\nPer-tool results:")
    print(_fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(_fmt_row(row))

    results_file = "inference_results.json"
    
    # Group results first by status, then by tool for easier viewing
    status_order = ["no_call", "wrong_tool", "wrong_params", "pass"]
    grouped_results = {status: {} for status in status_order}
    
    for result in results:
        status = result.get("status", "unknown")
        tool_name = result.get("expected_tool", "unknown")
        
        # Ensure status exists in grouped_results
        if status not in grouped_results:
            grouped_results[status] = {}
        
        # Group by tool within each status
        if tool_name not in grouped_results[status]:
            grouped_results[status][tool_name] = []
        grouped_results[status][tool_name].append(result)
    
    # Remove empty status groups
    grouped_results = {k: v for k, v in grouped_results.items() if v}
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(grouped_results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
