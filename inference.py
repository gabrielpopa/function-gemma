#!/usr/bin/env python3
"""
Inference script for fine-tuned FunctionGemma model.
Tests the model's ability to generate function calls for mobile actions.
"""

import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, set_seed


def _parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned FunctionGemma model.")
    parser.add_argument("--model_path", default="functiongemma-270m-it-mobile-actions-sft")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
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


def create_prompt(user_request, tools=None):
    """Create a properly formatted prompt for the model."""
    
    # Default mobile action tools
    if tools is None:
        tools = [
            {
                "function": {
                    "name": "turn_on_flashlight",
                    "description": "Turns the flashlight on.",
                    "parameters": {"type": "OBJECT", "properties": {}}
                }
            },
            {
                "function": {
                    "name": "turn_off_flashlight",
                    "description": "Turns the flashlight off.",
                    "parameters": {"type": "OBJECT", "properties": {}}
                }
            },
            {
                "function": {
                    "name": "create_contact",
                    "description": "Creates a contact in the phone's contact list.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "first_name": {"type": "STRING", "description": "The first name"},
                            "last_name": {"type": "STRING", "description": "The last name"},
                            "phone_number": {"type": "STRING", "description": "Phone number"},
                            "email": {"type": "STRING", "description": "Email address"}
                        },
                        "required": ["first_name", "last_name"]
                    }
                }
            },
            {
                "function": {
                    "name": "send_email",
                    "description": "Sends an email.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "to": {"type": "STRING", "description": "Recipient email"},
                            "subject": {"type": "STRING", "description": "Email subject"},
                            "body": {"type": "STRING", "description": "Email body"}
                        },
                        "required": ["to", "subject"]
                    }
                }
            },
            {
                "function": {
                    "name": "create_calendar_event",
                    "description": "Creates a new calendar event.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "title": {"type": "STRING", "description": "Event title"},
                            "datetime": {"type": "STRING", "description": "Date/time in YYYY-MM-DDTHH:MM:SS"}
                        },
                        "required": ["title", "datetime"]
                    }
                }
            },
            {
                "function": {
                    "name": "show_map",
                    "description": "Shows a location on the map.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING", "description": "Location to search"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "function": {
                    "name": "open_wifi_settings",
                    "description": "Opens the Wi-Fi settings.",
                    "parameters": {"type": "OBJECT", "properties": {}}
                }
            }
        ]
    
    return {
        "messages": [
            {
                "role": "developer",
                "content": "Current date and time: 2025-12-23T10:00:00\nYou are a model that can do function calling."
            },
            {
                "role": "user",
                "content": user_request
            }
        ],
        "tools": tools
    }


def test_model(model, tokenizer, processor, user_request, max_new_tokens, temperature):
    """Test the model with a user request."""
    
    print(f"\n{'='*60}")
    print(f"User Request: {user_request}")
    print(f"{'='*60}")
    
    # Create prompt
    prompt_data = create_prompt(user_request)
    
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
    
    # Test cases
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
