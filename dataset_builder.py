#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


PARAMETER_TYPES = ["STRING", "INTEGER", "NUMBER", "BOOLEAN", "ARRAY", "OBJECT"]


@dataclass
class DatasetStore:
    records: List[Dict[str, Any]] = field(default_factory=list)


def _prompt_choice(prompt: str, options: List[str]) -> str:
    print(prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}) {option}")
    while True:
        selection = input("Select an option: ").strip()
        if selection.isdigit() and 1 <= int(selection) <= len(options):
            return options[int(selection) - 1]
        print("Invalid selection. Try again.")


def _prompt_non_empty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Value cannot be empty.")


def _prompt_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    suffix = ""
    if default is True:
        suffix = " [Y/n]"
    elif default is False:
        suffix = " [y/N]"
    else:
        suffix = " [y/n]"
    while True:
        value = input(f"{prompt}{suffix}: ").strip().lower()
        if value == "" and default is not None:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"No existing file found at {path}. Starting fresh.")
        return []
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()
    if not content:
        return []
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    return json.loads(content)


def _save_json_or_jsonl(path: str, records: List[Dict[str, Any]], file_format: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if file_format == "jsonl":
        with open(path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")


def _load_tools(path: str) -> List[Dict[str, Any]]:
    tools = _load_json_or_jsonl(path)
    if not isinstance(tools, list):
        raise ValueError("Tools file must contain a list of tool definitions.")
    return tools


def _select_tool(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not tools:
        raise ValueError("No tools loaded.")
    tool_names = [tool["function"]["name"] for tool in tools]
    selected = _prompt_choice("Select a tool:", tool_names)
    return next(tool for tool in tools if tool["function"]["name"] == selected)


def _parse_value(raw: str, value_type: str, item_type: Optional[str] = None) -> Any:
    if value_type == "STRING":
        return raw
    if value_type == "INTEGER":
        return int(raw)
    if value_type == "NUMBER":
        return float(raw)
    if value_type == "BOOLEAN":
        return raw.lower() in {"true", "t", "1", "yes", "y"}
    if value_type == "ARRAY":
        if raw.strip().startswith("["):
            return json.loads(raw)
        items = [item.strip() for item in raw.split(",") if item.strip()]
        if item_type:
            return [_parse_value(item, item_type) for item in items]
        return items
    if value_type == "OBJECT":
        return json.loads(raw)
    return raw


def _build_arguments_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    args: Dict[str, Any] = {}
    for name, prop in properties.items():
        param_type = prop.get("type", "STRING")
        if param_type not in PARAMETER_TYPES:
            param_type = "STRING"
        if name in required:
            prompt = f"Enter value for required '{name}' ({param_type}): "
            raw = _prompt_non_empty(prompt)
        else:
            include = _prompt_yes_no(f"Include optional '{name}' ({param_type})?")
            if not include:
                continue
            raw = _prompt_non_empty(f"Enter value for '{name}' ({param_type}): ")
        item_type = None
        if param_type == "ARRAY":
            item_type = prop.get("items", {}).get("type", "STRING")
        try:
            args[name] = _parse_value(raw, param_type, item_type=item_type)
        except Exception:
            print("Could not parse value. Please enter a valid value (use JSON for OBJECT/ARRAY).")
            return _build_arguments_from_schema(schema)
    return args


def _collect_tool_arguments(tool: Dict[str, Any]) -> Dict[str, Any]:
    parameters = tool.get("function", {}).get("parameters", {})
    if not parameters:
        return {}
    use_json = _prompt_yes_no("Provide tool arguments as raw JSON?", default=False)
    if use_json:
        while True:
            raw = _prompt_non_empty("Arguments JSON object: ")
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
                print("Arguments must be a JSON object.")
            except json.JSONDecodeError:
                print("Invalid JSON. Try again.")
    return _build_arguments_from_schema(parameters)


def _default_developer_prompt() -> str:
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")
    day = now.strftime("%A")
    return (
        "Current date and time given in YYYY-MM-DDTHH:MM:SS format: "
        f"{timestamp}\nDay of week is {day}\n"
        "You are a model that can do function calling with the following functions\n"
    )


def _build_messages(
    tools: List[Dict[str, Any]], selected_tool: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    use_default = _prompt_yes_no(
        "Use default developer prompt with current date/time?", default=True
    )
    if use_default:
        developer_prompt = _default_developer_prompt()
    else:
        developer_prompt = _prompt_non_empty("Developer prompt: ")
    user_prompt = _prompt_non_empty("User prompt: ")
    tool = selected_tool or _select_tool(tools)
    arguments = _collect_tool_arguments(tool)
    return [
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": tool["function"]["name"], "arguments": arguments}}
            ],
        },
    ]


def _create_example(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = input("Metadata label [train]: ").strip() or "train"
    include_all = _prompt_yes_no("Include all tools in this record?", default=True)
    selected_tool = None
    if not include_all:
        selected_tool = _select_tool(tools)
    tools_to_use = tools if include_all else [selected_tool]
    messages = _build_messages(tools, selected_tool=selected_tool)
    return {"messages": messages, "tools": tools_to_use, "metadata": metadata}


def main() -> None:
    print("FunctionGemma Dataset Builder")
    tools_path = _prompt_non_empty("Path to tools file (json/jsonl): ")
    tools = _load_tools(tools_path)

    file_format = _prompt_choice("Choose dataset format:", ["json", "jsonl"])
    default_path = f"dataset.{file_format}"
    dataset_path = input(f"Enter dataset path to save/load [{default_path}]: ").strip() or default_path
    store = DatasetStore(records=_load_json_or_jsonl(dataset_path))

    while True:
        action = _prompt_choice(
            "\nWhat would you like to do?",
            [
                "Add a new example",
                "View dataset size",
                "Save and exit",
                "Exit without saving",
            ],
        )
        if action == "Add a new example":
            try:
                record = _create_example(tools)
            except ValueError as exc:
                print(exc)
                continue
            store.records.append(record)
            print("Added example.")
        elif action == "View dataset size":
            print(f"Dataset contains {len(store.records)} examples.")
        elif action == "Save and exit":
            _save_json_or_jsonl(dataset_path, store.records, file_format=file_format)
            print(f"Saved dataset to {dataset_path}.")
            break
        elif action == "Exit without saving":
            print("Exiting without saving.")
            break


if __name__ == "__main__":
    main()
