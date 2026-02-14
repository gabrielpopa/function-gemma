#!/usr/bin/env python3
import argparse
import json
import os
import sys
import textwrap
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
    instruction_role: str = "developer"
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
        if selection == "q":
            exit(0)
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


def _build_simple_export(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    exported: List[Dict[str, Any]] = []
    for record in records:
        user_content = _extract_user_prompt(record)
        tool_name = _extract_tool_name(record)
        arguments = _extract_arguments(record)
        exported.append(
            {
                "user_content": user_content,
                "tool_name": tool_name,
                "tool_arguments": json.dumps(arguments, ensure_ascii=False),
            }
        )
    return exported


def _save_simple_json(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exported = _build_simple_export(records)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(exported, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _load_tools(path: str) -> List[Dict[str, Any]]:
    tools = _load_json_or_jsonl(path)
    if not isinstance(tools, list):
        raise ValueError("Tools file must contain a list of tool definitions.")
    return tools


def _extract_tools_from_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools_map: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_tools = record.get("tools", [])
        if isinstance(record_tools, list):
            for tool in record_tools:
                name = tool.get("function", {}).get("name")
                if name:
                    tools_map[name] = tool
    return list(tools_map.values())


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
    default_instruction_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    instruction_role = settings.instruction_role if settings is not None else "developer"
    if settings is not None and settings.developer_prompt is not None:
        developer_prompt = settings.developer_prompt
    elif default_instruction_prompt:
        developer_prompt = default_instruction_prompt
    else:
        developer_prompt = _default_developer_prompt()
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
        {"role": instruction_role, "content": developer_prompt},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": tool["function"]["name"], "arguments": arguments}}
            ],
        },
    ]


def _create_example(
    tools: List[Dict[str, Any]],
    settings: ExampleSettings,
    default_instruction_prompt: Optional[str],
) -> Dict[str, Any]:
    metadata = input(f"Metadata label [{settings.metadata or 'train'}]: ").strip() or settings.metadata or "train"
    include_all = _prompt_yes_no(
        "Include all tools in this record?", default=settings.include_all_tools
    )
    settings.metadata = metadata
    settings.include_all_tools = include_all
    selected_tool = None
    if not include_all:
        selected_tool = _select_tool(tools, default_name=settings.selected_tool_name)
    tools_to_use = tools if include_all else [selected_tool]
    if selected_tool is not None:
        settings.selected_tool_name = selected_tool["function"]["name"]
    messages = _build_messages(
        tools,
        selected_tool=selected_tool,
        settings=settings,
        default_instruction_prompt=default_instruction_prompt,
    )
    return {"messages": messages, "tools": tools_to_use, "metadata": metadata}


def _create_example_with_settings(
    tools: List[Dict[str, Any]],
    settings: ExampleSettings,
    default_instruction_prompt: Optional[str],
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
        default_instruction_prompt=default_instruction_prompt,
    )
    return {"messages": messages, "tools": tools_to_use, "metadata": settings.metadata}


def _infer_format_from_path(path: str, default: str = "jsonl") -> str:
    if path.endswith(".jsonl"):
        return "jsonl"
    if path.endswith(".json"):
        return "json"
    return default


def _infer_instruction_from_records(
    records: List[Dict[str, Any]],
) -> (Optional[str], Optional[str]):
    for record in reversed(records):
        messages = record.get("messages", [])
        if not isinstance(messages, list) or not messages:
            continue
        first = messages[0]
        role = first.get("role")
        content = first.get("content")
        if role and isinstance(content, str):
            return role, content
    return None, None


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


def _extract_roles(record: Dict[str, Any]) -> str:
    roles = []
    for msg in record.get("messages", []):
        role = msg.get("role")
        if role and role not in roles:
            roles.append(role)
    return ", ".join(roles)


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
        roles = _extract_roles(record)
        arguments = _extract_arguments(record)
        rows.append(
            [metadata, roles, user_prompt, json.dumps(arguments, ensure_ascii=False)]
        )

    if not rows:
        print("No matching entries.")
        return tool_name, split

    headers = ["Split", "Roles", "User Prompt", "Arguments"]
    max_widths = [8, 18, 60, 60]

    def _wrap_cell(text: str, width: int) -> List[str]:
        if not text:
            return [""]
        return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]

    def _print_table(rows_in: List[List[str]]) -> None:
        col_widths = [min(len(h), max_widths[i]) for i, h in enumerate(headers)]
        for row in rows_in:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], min(len(str(cell)), max_widths[i]))

        def _fmt_line(values: List[str]) -> str:
            return " | ".join(str(values[i]).ljust(col_widths[i]) for i in range(len(values)))

        print(_fmt_line(headers))
        print("-+-".join("-" * w for w in col_widths))
        for row in rows_in:
            wrapped_cells = [
                _wrap_cell(str(row[i]), col_widths[i]) for i in range(len(row))
            ]
            max_lines = max(len(cell_lines) for cell_lines in wrapped_cells)
            for line_idx in range(max_lines):
                line_values = [
                    (cell_lines[line_idx] if line_idx < len(cell_lines) else "")
                    for cell_lines in wrapped_cells
                ]
                print(_fmt_line(line_values))

    _print_table(rows)
    return tool_name, split


def _configure_instruction_settings(
    settings: ExampleSettings, default_instruction_prompt: Optional[str]
) -> None:
    settings.instruction_role = _prompt_choice(
        "Instruction role:",
        ["developer", "system", "user"],
        default=settings.instruction_role,
    )
    current_prompt = settings.developer_prompt
    if current_prompt is None:
        current_prompt = default_instruction_prompt or _default_developer_prompt()
    print("\nCurrent instruction prompt:\n")
    print(current_prompt)
    new_prompt = input(
        "\nEnter new instruction prompt (leave empty to keep current, type CLEAR to reset to default): "
    ).strip()
    if not new_prompt:
        return
    if new_prompt.upper() == "CLEAR":
        settings.developer_prompt = None
        return
    settings.developer_prompt = new_prompt


def _delete_entry_for_tool(
    records: List[Dict[str, Any]], tools: List[Dict[str, Any]], last_tool_name: Optional[str]
) -> Optional[str]:
    tool = _select_tool(tools, default_name=last_tool_name)
    tool_name = tool["function"]["name"]
    matching_indices = [
        idx for idx, record in enumerate(records) if _extract_tool_name(record) == tool_name
    ]
    if not matching_indices:
        print("No entries found for that tool.")
        return tool_name
    recent_indices = list(reversed(matching_indices))[:10]
    rows = []
    for display_idx, record_idx in enumerate(recent_indices, start=1):
        record = records[record_idx]
        metadata = str(record.get("metadata", ""))
        user_prompt = _extract_user_prompt(record)
        arguments = json.dumps(_extract_arguments(record), ensure_ascii=False)
        rows.append([str(display_idx), metadata, user_prompt, arguments])

    headers = ["#", "Split", "User Prompt", "Arguments"]
    max_widths = [3, 8, 60, 60]

    def _wrap_cell(text: str, width: int) -> List[str]:
        if not text:
            return [""]
        return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]

    def _print_table(rows_in: List[List[str]]) -> None:
        col_widths = [min(len(h), max_widths[i]) for i, h in enumerate(headers)]
        for row in rows_in:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], min(len(str(cell)), max_widths[i]))

        def _fmt_line(values: List[str]) -> str:
            return " | ".join(str(values[i]).ljust(col_widths[i]) for i in range(len(values)))

        print(_fmt_line(headers))
        print("-+-".join("-" * w for w in col_widths))
        for row in rows_in:
            wrapped_cells = [
                _wrap_cell(str(row[i]), col_widths[i]) for i in range(len(row))
            ]
            max_lines = max(len(cell_lines) for cell_lines in wrapped_cells)
            for line_idx in range(max_lines):
                line_values = [
                    (cell_lines[line_idx] if line_idx < len(cell_lines) else "")
                    for cell_lines in wrapped_cells
                ]
                print(_fmt_line(line_values))

    _print_table(rows)
    while True:
        selection = input("Select entry number to delete (0 to cancel): ").strip()
        if selection == "0":
            print("Delete cancelled.")
            return tool_name
        if selection.isdigit():
            num = int(selection)
            if 1 <= num <= len(recent_indices):
                record_index = recent_indices[num - 1]
                del records[record_index]
                print("Entry deleted.")
                return tool_name
        print("Invalid selection. Try again.")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive builder for FunctionGemma datasets."
    )
    parser.add_argument(
        "--tools",
        default="",
        help="Path to tools file (json/jsonl). If omitted, tools are extracted from dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="dataset.jsonl",
        help="Path to dataset file (json/jsonl). Default: dataset.jsonl",
    )
    parser.add_argument(
        "--tool-name",
        "--tool",
        dest="tool_name",
        default="",
        help="Tool name to use when adding a single example via CLI.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="User prompt to use when adding a single example via CLI.",
    )
    parser.add_argument(
        "--metadata",
        default="",
        help="Metadata label for the record (default: train).",
    )
    parser.add_argument(
        "--instruction-role",
        default="",
        help="Role for the instruction message (developer/system/user). Default: last used.",
    )
    parser.add_argument(
        "--instruction-prompt",
        default="",
        help="Instruction message content. Default: last used from dataset.",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Tool argument in key=value form. Can be repeated.",
    )
    parser.add_argument(
        "--arg-json",
        default="",
        help="Tool arguments as a JSON object (merged with --arg).",
    )
    parser.add_argument(
        "--only-tool",
        action="store_true",
        help="Include only the selected tool (instead of all tools).",
    )
    parser.add_argument(
        "--export-simple",
        default="",
        help=(
            "Export a simplified JSON dataset (user_content/tool_name/tool_arguments) "
            "to the provided path and exit. Works with or without CLI add mode."
        ),
    )
    return parser.parse_args(argv)


def _coerce_arg_value(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return raw
    if raw[0] in "-0123456789tfn[{\"":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _parse_cli_tool_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {}
    if args.arg_json:
        try:
            parsed = json.loads(args.arg_json)
            if not isinstance(parsed, dict):
                raise ValueError("--arg-json must be a JSON object.")
            arguments.update(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for --arg-json: {exc}") from exc
    for item in args.arg:
        if "=" not in item:
            raise ValueError(f"Invalid --arg '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --arg '{item}'. Key cannot be empty.")
        arguments[key] = _coerce_arg_value(value)
    return arguments


def _validate_required_args(tool: Dict[str, Any], arguments: Dict[str, Any]) -> None:
    parameters = tool.get("function", {}).get("parameters", {})
    required = parameters.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")


def _build_record_from_cli(
    tools: List[Dict[str, Any]],
    args: argparse.Namespace,
    default_instruction_role: Optional[str],
    default_instruction_prompt: Optional[str],
) -> Dict[str, Any]:
    if not args.tool_name:
        raise ValueError("Missing --tool-name/--tool.")
    if not args.prompt:
        raise ValueError("Missing --prompt.")
    tool = next(
        (t for t in tools if t.get("function", {}).get("name") == args.tool_name),
        None,
    )
    if tool is None:
        raise ValueError(f"Tool '{args.tool_name}' not found.")
    metadata = args.metadata or "train"
    arguments = _parse_cli_tool_arguments(args)
    _validate_required_args(tool, arguments)
    tools_to_use = [tool] if args.only_tool else tools
    instruction_role = args.instruction_role or default_instruction_role or "developer"
    if args.instruction_prompt:
        instruction_prompt = args.instruction_prompt
    elif default_instruction_prompt:
        instruction_prompt = default_instruction_prompt
    else:
        instruction_prompt = _default_developer_prompt()
    messages = [
        {"role": instruction_role, "content": instruction_prompt},
        {"role": "user", "content": args.prompt},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": tool["function"]["name"], "arguments": arguments}}
            ],
        },
    ]
    return {"messages": messages, "tools": tools_to_use, "metadata": metadata}


def main() -> None:
    print("FunctionGemma Dataset Builder")
    args = _parse_args(sys.argv[1:])
    tools_path = args.tools

    cli_add_mode = bool(
        args.tool_name
        or args.prompt
        or args.metadata
        or args.arg
        or args.arg_json
        or args.only_tool
        or args.instruction_role
        or args.instruction_prompt
    )
    dataset_path = args.dataset
    if not cli_add_mode and "--dataset" not in sys.argv[1:]:
        dataset_path = input(f"Enter dataset path to save/load [{dataset_path}]: ").strip() or dataset_path
    file_format = _infer_format_from_path(dataset_path)
    store = DatasetStore(records=_load_json_or_jsonl(dataset_path))
    default_instruction_role, default_instruction_prompt = _infer_instruction_from_records(
        store.records
    )
    tools: List[Dict[str, Any]] = []
    if tools_path:
        try:
            tools = _load_tools(tools_path)
        except (ValueError, KeyError):
            tools = []
    if not tools:
        tools = _extract_tools_from_records(store.records)
    if not tools:
        print("No tools found. Provide --tools or add tools to the dataset records.")
        if cli_add_mode:
            sys.exit(1)
    if cli_add_mode:
        try:
            record = _build_record_from_cli(
                tools, args, default_instruction_role, default_instruction_prompt
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        store.records.append(record)
        _save_json_or_jsonl(dataset_path, store.records, file_format=file_format)
        print(f"Saved dataset to {dataset_path}.")
        if args.export_simple:
            _save_simple_json(args.export_simple, store.records)
            print(f"Exported simplified dataset to {args.export_simple}.")
        return
    if args.export_simple:
        _save_simple_json(args.export_simple, store.records)
        print(f"Exported simplified dataset to {args.export_simple}.")
        return
    last_settings: ExampleSettings = ExampleSettings(
        metadata="train",
        include_all_tools=True,
        selected_tool_name=None,
        instruction_role=default_instruction_role or "developer",
        developer_prompt=None,
        use_json_arguments=False,
    )
    last_tool_name: Optional[str] = None
    last_split: Optional[str] = None

    while True:
        action = _prompt_choice(
            "\nWhat would you like to do?\n",
            [
                "Add a new example",
                "Add a new example for last used tool",
                "View dataset size",
                "Show dataset for a tool",
                "Set instruction role/prompt",
                "Delete entry for a tool",
                "Export simplified JSON",
                "Save and exit",
                "Exit without saving\n",
            ],
        )
        if action == "Add a new example":
            try:
                record = _create_example(tools, last_settings, default_instruction_prompt)
            except ValueError as exc:
                print(exc)
                continue
            store.records.append(record)
            last_tool_name = last_settings.selected_tool_name
            last_split = last_settings.metadata
            default_instruction_role, default_instruction_prompt = _infer_instruction_from_records(
                store.records
            )
            print("Added example.")
        elif action == "Add a new example for last used tool":
            try:
                record = _create_example_with_settings(
                    tools, last_settings, default_instruction_prompt
                )
            except ValueError as exc:
                print(exc)
                continue
            store.records.append(record)
            last_tool_name = last_settings.selected_tool_name
            last_split = last_settings.metadata
            default_instruction_role, default_instruction_prompt = _infer_instruction_from_records(
                store.records
            )
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
        elif action == "Set instruction role/prompt":
            _configure_instruction_settings(last_settings, default_instruction_prompt)
        elif action == "Delete entry for a tool":
            try:
                last_tool_name = _delete_entry_for_tool(
                    store.records, tools, last_tool_name
                )
                default_instruction_role, default_instruction_prompt = _infer_instruction_from_records(
                    store.records
                )
            except ValueError as exc:
                print(exc)
        elif action == "Export simplified JSON":
            default_export_path = "dataset.json"
            export_path = input(
                f"Enter export path [{default_export_path}]: "
            ).strip() or default_export_path
            _save_simple_json(export_path, store.records)
            print(f"Exported simplified dataset to {export_path}.")
        elif action == "Save and exit":
            _save_json_or_jsonl(dataset_path, store.records, file_format=file_format)
            print(f"Saved dataset to {dataset_path}.")
            break
        elif action.startswith("Exit without saving"):
            if _prompt_yes_no("Exit without saving?", default=False):
                print("Exiting without saving.")
                break


if __name__ == "__main__":
    main()
