#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


PARAMETER_TYPES = ["STRING", "INTEGER", "NUMBER", "BOOLEAN", "ARRAY", "OBJECT"]


@dataclass
class DatasetStore:
    records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExampleSettings:
    metadata: str = "train"
    include_all_tools: bool = True
    selected_tool_name: Optional[str] = None
    use_default_developer_prompt: bool = True
    developer_prompt: Optional[str] = None
    use_json_arguments: bool = False


def _prompt_choice(
    prompt: str, options: List[str], default: Optional[str] = None
) -> str:
    print(prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}) {option}")
    while True:
        default_hint = f" [{default}]" if default else ""
        selection = input(f"Select an option{default_hint}: ").strip()
        if selection == "" and default:
            return default
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


def _select_tool(
    tools: List[Dict[str, Any]], default_name: Optional[str] = None
) -> Dict[str, Any]:
    if not tools:
        raise ValueError("No tools loaded.")
    tool_names = [tool["function"]["name"] for tool in tools]
    selected = _prompt_choice("Select a tool:", tool_names, default=default_name)
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
            include = _prompt_yes_no(
                f"Include optional '{name}' ({param_type})?", default=False
            )
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


def _collect_tool_arguments(
    tool: Dict[str, Any], settings: Optional[ExampleSettings] = None
) -> Dict[str, Any]:
    parameters = tool.get("function", {}).get("parameters", {})
    if not parameters:
        return {}
    default_use_json = settings.use_json_arguments if settings is not None else False
    use_json = _prompt_yes_no(
        "Provide tool arguments as raw JSON?", default=default_use_json
    )
    if settings is not None:
        settings.use_json_arguments = use_json
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
    tools: List[Dict[str, Any]],
    selected_tool: Optional[Dict[str, Any]] = None,
    settings: Optional[ExampleSettings] = None,
    skip_developer_prompt: bool = False,
) -> List[Dict[str, Any]]:
    if skip_developer_prompt and settings is not None:
        if settings.use_default_developer_prompt:
            developer_prompt = _default_developer_prompt()
        else:
            developer_prompt = settings.developer_prompt or _prompt_non_empty("Developer prompt: ")
    else:
        use_default = _prompt_yes_no(
            "Use default developer prompt with current date/time?", default=True
        )
        if use_default:
            developer_prompt = _default_developer_prompt()
        else:
            developer_prompt = _prompt_non_empty("Developer prompt: ")
            if settings is not None:
                settings.use_default_developer_prompt = False
                settings.developer_prompt = developer_prompt
    tool = selected_tool
    if tool is None and settings is not None and settings.selected_tool_name:
        tool = next(
            (t for t in tools if t["function"]["name"] == settings.selected_tool_name),
            None,
        )
    if tool is None:
        tool = _select_tool(tools)
    if settings is not None:
        settings.selected_tool_name = tool["function"]["name"]
    user_prompt = _prompt_non_empty("User prompt: ")
    arguments = _collect_tool_arguments(tool, settings=settings)
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


def _create_example(tools: List[Dict[str, Any]]) -> (Dict[str, Any], ExampleSettings):
    metadata = input("Metadata label [train]: ").strip() or "train"
    include_all = _prompt_yes_no("Include all tools in this record?", default=True)
    selected_tool = None
    if not include_all:
        selected_tool = _select_tool(tools)
    tools_to_use = tools if include_all else [selected_tool]
    settings = ExampleSettings(
        metadata=metadata,
        include_all_tools=include_all,
        selected_tool_name=selected_tool["function"]["name"] if selected_tool else None,
    )
    messages = _build_messages(tools, selected_tool=selected_tool, settings=settings)
    return {"messages": messages, "tools": tools_to_use, "metadata": metadata}, settings


def _create_example_with_settings(
    tools: List[Dict[str, Any]], settings: ExampleSettings
) -> Dict[str, Any]:
    selected_tool = None
    if not settings.include_all_tools and settings.selected_tool_name:
        selected_tool = next(
            (tool for tool in tools if tool["function"]["name"] == settings.selected_tool_name),
            None,
        )
        if selected_tool is None:
            raise ValueError(f"Tool '{settings.selected_tool_name}' not found.")
    tools_to_use = tools if settings.include_all_tools else [selected_tool]
    messages = _build_messages(
        tools,
        selected_tool=selected_tool,
        settings=settings,
        skip_developer_prompt=True,
    )
    return {"messages": messages, "tools": tools_to_use, "metadata": settings.metadata}


def _infer_format_from_path(path: str, default: str = "jsonl") -> str:
    if path.endswith(".jsonl"):
        return "jsonl"
    if path.endswith(".json"):
        return "json"
    return default


def _print_dataset_stats(records: List[Dict[str, Any]]) -> None:
    total = len(records)
    meta_counts: Dict[str, int] = {}
    tool_counts: Dict[str, Dict[str, int]] = {}

    for record in records:
        metadata = str(record.get("metadata", ""))
        meta_counts[metadata] = meta_counts.get(metadata, 0) + 1

        tool_name = "(unknown)"
        for msg in record.get("messages", []):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                call = msg["tool_calls"][0]
                tool_name = call.get("function", {}).get("name") or tool_name
                break

        if tool_name not in tool_counts:
            tool_counts[tool_name] = {"total": 0, "train": 0, "eval": 0, "other": 0}
        tool_counts[tool_name]["total"] += 1
        if metadata == "train":
            tool_counts[tool_name]["train"] += 1
        elif metadata == "eval":
            tool_counts[tool_name]["eval"] += 1
        else:
            tool_counts[tool_name]["other"] += 1

    print(f"Dataset contains {total} examples.")
    if meta_counts:
        print("By metadata:")
        for key, count in sorted(meta_counts.items()):
            label = key if key else "(empty)"
            print(f"- {label}: {count}")

    if not tool_counts:
        return
    headers = ["Tool", "Total", "Train", "Eval", "Other"]
    rows = []
    for tool_name, counts in sorted(tool_counts.items()):
        rows.append(
            [
                tool_name,
                str(counts["total"]),
                str(counts["train"]),
                str(counts["eval"]),
                str(counts["other"]),
            ]
        )
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def _fmt_row(values: List[str]) -> str:
        return " | ".join(str(values[i]).ljust(col_widths[i]) for i in range(len(values)))

    print("\nBy tool:")
    print(_fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(_fmt_row(row))


def _explain_train_eval() -> None:
    print("Train vs eval:")
    print("- train: examples used to update model weights during fine-tuning.")
    print("- eval: examples held out for evaluation/validation; not used for updates.")
    print("- other: any metadata label that is not 'train' or 'eval'.")


def _extract_tool_name(record: Dict[str, Any]) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            call = msg["tool_calls"][0]
            name = call.get("function", {}).get("name")
            if name:
                return name
    return "(unknown)"


def _extract_user_prompt(record: Dict[str, Any]) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _extract_arguments(record: Dict[str, Any]) -> Dict[str, Any]:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            call = msg["tool_calls"][0]
            return call.get("function", {}).get("arguments", {})
    return {}


def _show_dataset_for_tool(
    records: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    last_tool_name: Optional[str],
    last_split: Optional[str],
) -> (Optional[str], Optional[str]):
    tool = _select_tool(tools, default_name=last_tool_name)
    tool_name = tool["function"]["name"]
    split_default = last_split or "all"
    split = _prompt_choice("Choose split:", ["train", "eval", "all"], default=split_default)
    rows = []
    for record in records:
        metadata = str(record.get("metadata", ""))
        if split != "all" and metadata != split:
            continue
        if _extract_tool_name(record) != tool_name:
            continue
        user_prompt = _extract_user_prompt(record)
        arguments = _extract_arguments(record)
        rows.append([metadata, user_prompt, json.dumps(arguments, ensure_ascii=False)])

    if not rows:
        print("No matching entries.")
        return tool_name, split

    headers = ["Split", "User Prompt", "Arguments"]
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def _fmt_row(values: List[str]) -> str:
        return " | ".join(str(values[i]).ljust(col_widths[i]) for i in range(len(values)))

    print(_fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(_fmt_row(row))
    return tool_name, split


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive builder for FunctionGemma datasets."
    )
    parser.add_argument(
        "--tools",
        default="tools.jsonl",
        help="Path to tools file (json/jsonl). Default: tools.jsonl",
    )
    parser.add_argument(
        "--dataset",
        default="dataset.jsonl",
        help="Path to dataset file (json/jsonl). Default: dataset.jsonl",
    )
    return parser.parse_args(argv)


def main() -> None:
    print("FunctionGemma Dataset Builder")
    args = _parse_args(sys.argv[1:])
    tools_path = args.tools
    tools = _load_tools(tools_path)

    dataset_path = args.dataset
    if "--dataset" not in sys.argv[1:]:
        dataset_path = input(f"Enter dataset path to save/load [{dataset_path}]: ").strip() or dataset_path
    file_format = _infer_format_from_path(dataset_path)
    store = DatasetStore(records=_load_json_or_jsonl(dataset_path))
    last_settings: Optional[ExampleSettings] = None
    last_tool_name: Optional[str] = None
    last_split: Optional[str] = None

    while True:
        action = _prompt_choice(
            "\nWhat would you like to do?",
            [
                "Add a new example",
                "Add a new example for last used tool",
                "View dataset size",
                "Show dataset for a tool",
                "Save and exit",
                "Exit without saving",
            ],
        )
        if action == "Add a new example":
            try:
                record, last_settings = _create_example(tools)
            except ValueError as exc:
                print(exc)
                continue
            store.records.append(record)
            if last_settings is not None:
                last_tool_name = last_settings.selected_tool_name
                last_split = last_settings.metadata
            print("Added example.")
        elif action == "Add a new example for last used tool":
            if last_settings is None:
                print("No previous example settings yet.")
                continue
            try:
                record = _create_example_with_settings(tools, last_settings)
            except ValueError as exc:
                print(exc)
                continue
            store.records.append(record)
            last_tool_name = last_settings.selected_tool_name
            last_split = last_settings.metadata
            print("Added example.")
        elif action == "View dataset size":
            _print_dataset_stats(store.records)
        elif action == "Show dataset for a tool":
            try:
                tool_name, split = _show_dataset_for_tool(
                    store.records, tools, last_tool_name, last_split
                )
                last_tool_name = tool_name
                last_split = split
            except ValueError as exc:
                print(exc)
        elif action == "Save and exit":
            _save_json_or_jsonl(dataset_path, store.records, file_format=file_format)
            print(f"Saved dataset to {dataset_path}.")
            break
        elif action == "Exit without saving":
            if _prompt_yes_no("Exit without saving?"):
                print("Exiting without saving.")
                break


if __name__ == "__main__":
    main()
