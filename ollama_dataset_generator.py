#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_EXPLANATION = (
    "Generate a single example user message for this tool.\n"
    "Reply ONLY with a single-line JSON object: {\"message\": \"...\"}\n"
    "Do not include any extra text, explanations, or code fences.\n"
)


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()
    if not content:
        return []
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    data = json.loads(content)
    if isinstance(data, list):
        return data
    return []


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        value = input(f"{prompt}{suffix}: ").strip().lower()
        if value == "" and default is not None:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _load_tools_from_dataset(path: str) -> List[Dict[str, Any]]:
    records = _load_json_or_jsonl(path)
    tools_map: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_tools = record.get("tools", [])
        if isinstance(record_tools, list):
            for tool in record_tools:
                name = tool.get("function", {}).get("name")
                if name:
                    tools_map[name] = tool
    return list(tools_map.values())


def _load_tools_from_file(path: str) -> List[Dict[str, Any]]:
    data = _load_json_or_jsonl(path)
    if not isinstance(data, list):
        raise ValueError("Tools file must contain a list of tool definitions.")
    return data


def _find_tool(tools: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for tool in tools:
        if tool.get("function", {}).get("name") == name:
            return tool
    raise ValueError(f"Tool '{name}' not found.")


def _build_step1_prompt(
    tool: Dict[str, Any], explanation: Optional[str], examples: List[str]
) -> str:
    payload = {
        "name": tool.get("function", {}).get("name", ""),
        "description": tool.get("function", {}).get("description", ""),
    }
    parts = [DEFAULT_EXPLANATION]
    if explanation:
        parts.append("Additional guidance:")
        parts.append(explanation)
    parts.extend(
        [
            "Make it different from the examples and vary object/target.",
            "Tool info:",
            json.dumps(payload, indent=2),
        ]
    )
    if examples:
        parts.append("Recent user examples (do not copy verbatim):")
        for ex in examples:
            parts.append(f"- {ex}")
    return "\n".join(parts)


def _build_step2_prompt(
    tool: Dict[str, Any],
    user_message: str,
    include_optional: bool,
    examples: List[Tuple[str, Dict[str, Any]]],
) -> str:
    payload = {
        "name": tool.get("function", {}).get("name", ""),
        "description": tool.get("function", {}).get("description", ""),
        "parameters": tool.get("function", {}).get("parameters", {}),
    }
    required = _required_args(tool)
    required_note = (
        "Always include all required fields: " + ", ".join(required) if required else "No required fields."
    )
    optional_note = (
        "Include ALL optional fields with plausible values."
        if include_optional
        else "Omit optional fields if not explicitly present."
    )
    parts = [
        "Extract tool arguments from the user message.",
        "Return ONLY a single-line JSON object of arguments and nothing else.",
        required_note,
        optional_note,
        "Tool schema:",
        json.dumps(payload, indent=2),
    ]
    if examples:
        parts.append("Examples (message -> arguments). Use for guidance only:")
        for message, args in examples:
            parts.append(f"- message: {message}")
            parts.append(f"  arguments: {json.dumps(args, ensure_ascii=False)}")
    parts.extend(
        [
        "User message:",
        user_message,
        ]
    )
    return "\n".join(parts)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    # Prefer a single-line JSON object if present.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != -1 and end != -1 and end > start:
            inner = text[start + 3 : end].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        snippet = text[left : right + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _extract_first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _extract_message_json(text: str) -> str:
    parsed = _extract_json_from_text(text)
    if not parsed:
        return ""
    value = parsed.get("message")
    if isinstance(value, str):
        return value.strip()
    return ""


def _clean_user_message(text: str) -> str:
    msg = text.strip()
    if len(msg) >= 2 and ((msg[0] == msg[-1]) and msg[0] in {"'", '"'}):
        msg = msg[1:-1].strip()
    return msg


def _looks_like_bad_example(text: str) -> bool:
    lowered = text.lower()
    if "here's an example" in lowered or "sure!" in lowered:
        return True
    if lowered.endswith(":"):
        return True
    return False


def _record_tool_name(record: Dict[str, Any]) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            call = msg["tool_calls"][0]
            name = call.get("function", {}).get("name")
            if name:
                return name
    return ""


def _record_user_message(record: Dict[str, Any]) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _record_arguments(record: Dict[str, Any]) -> Dict[str, Any]:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            call = msg["tool_calls"][0]
            args = call.get("function", {}).get("arguments", {})
            if isinstance(args, dict):
                return args
    return {}


def _recent_examples(
    records: List[Dict[str, Any]], tool_name: str, max_examples: int = 10
) -> List[str]:
    examples: List[str] = []
    for record in records:
        if _record_tool_name(record) != tool_name:
            continue
        user_msg = _clean_user_message(_record_user_message(record))
        if user_msg and not _looks_like_bad_example(user_msg):
            examples.append(user_msg)
    if not examples:
        return []
    if len(examples) <= max_examples:
        return examples
    return random.sample(examples, k=max_examples)


def _argument_examples(
    records: List[Dict[str, Any]], tool_name: str, max_examples: int = 10
) -> List[Tuple[str, Dict[str, Any]]]:
    examples: List[Tuple[str, Dict[str, Any]]] = []
    for record in records:
        if _record_tool_name(record) != tool_name:
            continue
        user_msg = _clean_user_message(_record_user_message(record))
        if not user_msg or _looks_like_bad_example(user_msg):
            continue
        arguments = _record_arguments(record)
        if not isinstance(arguments, dict):
            continue
        examples.append((user_msg, arguments))
    if not examples:
        return []
    if len(examples) <= max_examples:
        return examples
    return random.sample(examples, k=max_examples)


def _required_args(tool: Dict[str, Any]) -> List[str]:
    params = tool.get("function", {}).get("parameters", {})
    required = params.get("required", [])
    return [str(name) for name in required]


def _validate_arguments(tool: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
    required = _required_args(tool)
    missing = [name for name in required if name not in arguments]
    return (len(missing) == 0), missing


def _run_ollama(model: str, prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ollama run failed")
    return result.stdout.strip()


def _call_dataset_builder(
    dataset_path: str,
    tool_name: str,
    prompt: str,
    arguments: Dict[str, Any],
) -> None:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "dataset_builder.py"),
        "--dataset",
        dataset_path,
        "--tool",
        tool_name,
        "--prompt",
        prompt,
        "--arg-json",
        json.dumps(arguments, ensure_ascii=False),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or result.stderr.strip() or "dataset_builder failed")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset entries using local Ollama and append via dataset_builder.py."
    )
    parser.add_argument("--ollama-model", required=True, help="Ollama model name.")
    parser.add_argument("--dataset", required=True, help="Dataset JSON/JSONL file.")
    parser.add_argument("--tool", required=True, help="Tool name to generate entries for.")
    parser.add_argument(
        "--max-entries", type=int, default=10, help="Number of entries to generate (default: 10)."
    )
    parser.add_argument(
        "--explanation",
        default="",
        help="Explanation prompt for Ollama (overrides default).",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Ask Ollama to include all optional tool fields.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per entry if the generated user message repeats (default: 3).",
    )
    parser.add_argument(
        "--tools",
        default="",
        help="Optional tools file (json/jsonl). If omitted, tools are loaded from dataset.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Ask before inserting each generated entry.",
    )
    parser.add_argument(
        "--auto-insert",
        action="store_true",
        help="Insert entries without asking for confirmation (overrides --confirm).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the Ollama prompt and raw responses for debugging.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])

    if args.max_entries <= 0:
        print("max-entries must be > 0")
        sys.exit(1)
    if args.auto_insert:
        args.confirm = False

    dataset_records = _load_json_or_jsonl(args.dataset)
    if args.tools:
        tools = _load_tools_from_file(args.tools)
    else:
        tools = _load_tools_from_dataset(args.dataset)
    if not tools:
        print("No tools found. Provide --tools or ensure the dataset contains tools.")
        sys.exit(1)

    tool = _find_tool(tools, args.tool)
    explanation = args.explanation.strip() or DEFAULT_EXPLANATION
    examples = _recent_examples(dataset_records, args.tool, max_examples=10)
    argument_examples = _argument_examples(dataset_records, args.tool, max_examples=10)
    prompt_template = _build_step1_prompt(tool, explanation, examples)
    seen_messages = {ex.strip().lower() for ex in examples if ex.strip()}

    if args.debug:
        print("Ollama step 1 prompt:")
        print(prompt_template)
        print("-" * 40)

    successes = 0
    for idx in range(args.max_entries):
        user_message = ""
        for attempt in range(args.max_retries + 1):
            try:
                output = _run_ollama(args.ollama_model, prompt_template)
            except RuntimeError as exc:
                print(f"[{idx + 1}] Ollama error: {exc}")
                output = ""
            if args.debug:
                print(f"[{idx + 1}] Raw Ollama step 1 output:")
                print(output)
                print("-" * 40)
            user_message = _extract_message_json(output)
            if not user_message:
                if attempt >= args.max_retries:
                    break
                continue
            if user_message.strip().lower() in seen_messages:
                if attempt >= args.max_retries:
                    break
                prompt_template = (
                    prompt_template
                    + "\nDo NOT repeat any previous example. Use different object and target.\n"
                )
                continue
            break
        if not user_message:
            print(f"[{idx + 1}] Empty or repeated user message. Skipping.")
            continue
        seen_messages.add(user_message.strip().lower())

        step2_prompt = _build_step2_prompt(
            tool, user_message, args.include_optional, argument_examples
        )
        if args.debug:
            print(f"[{idx + 1}] Ollama step 2 prompt:")
            print(step2_prompt)
            print("-" * 40)

        try:
            output2 = _run_ollama(args.ollama_model, step2_prompt)
        except RuntimeError as exc:
            print(f"[{idx + 1}] Ollama error (step 2): {exc}")
            continue

        if args.debug:
            print(f"[{idx + 1}] Raw Ollama step 2 output:")
            print(output2)
            print("-" * 40)

        arguments = _extract_json_from_text(output2)
        if not arguments:
            print(f"[{idx + 1}] Could not parse arguments JSON. Skipping.")
            continue
        if not isinstance(arguments, dict):
            print(f"[{idx + 1}] Arguments must be a JSON object. Skipping.")
            continue

        ok, missing = _validate_arguments(tool, arguments)
        if not ok:
            print(f"[{idx + 1}] Missing required args: {', '.join(missing)}. Skipping.")
            continue

        if args.confirm:
            color_msg = "\033[1;32m"  # bright green
            color_args = "\033[1;36m"  # bright cyan
            reset = "\033[0m"
            print(f"User message:")
            print(f"{color_msg}{user_message.strip()}{reset}")
            print(f"Arguments:")
            print(f"{color_args}{json.dumps(arguments, ensure_ascii=False)}{reset}")
            if not _prompt_yes_no("Insert into dataset?", default=True):
                print(f"[{idx + 1}] Skipped.")
                continue

        try:
            _call_dataset_builder(args.dataset, args.tool, user_message.strip(), arguments)
        except RuntimeError as exc:
            print(f"[{idx + 1}] dataset_builder error: {exc}")
            continue

        successes += 1
        print(f"[{idx + 1}] Added entry.")

    print(f"Completed. Added {successes}/{args.max_entries} entries.")


if __name__ == "__main__":
    main()
