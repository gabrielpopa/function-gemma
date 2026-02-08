#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


PARAMETER_TYPES = ["STRING", "INTEGER", "NUMBER", "BOOLEAN", "ARRAY", "OBJECT"]


@dataclass
class ToolStore:
    tools: List[Dict[str, Dict[str, object]]] = field(default_factory=list)

    def find_tool(self, name: str) -> Optional[Dict[str, Dict[str, object]]]:
        return next((tool for tool in self.tools if tool["function"]["name"] == name), None)

    def add_tool(self, name: str, description: str) -> None:
        self.tools.append(
            {
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "OBJECT", "properties": {}, "required": []},
                }
            }
        )

    def add_parameter(
        self,
        tool_name: str,
        param_name: str,
        param_type: str,
        description: str,
        required: bool,
        array_item_type: Optional[str],
    ) -> None:
        tool = self.find_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        parameters = tool["function"]["parameters"]
        properties = parameters.setdefault("properties", {})
        schema: Dict[str, object] = {"type": param_type}
        if description:
            schema["description"] = description
        if param_type == "ARRAY":
            schema["items"] = {"type": array_item_type or "STRING"}
        properties[param_name] = schema
        if required:
            required_list = parameters.setdefault("required", [])
            if param_name not in required_list:
                required_list.append(param_name)

    def delete_tool(self, name: str) -> bool:
        before = len(self.tools)
        self.tools = [tool for tool in self.tools if tool["function"]["name"] != name]
        return len(self.tools) != before

    def delete_parameter(self, tool_name: str, param_name: str) -> bool:
        tool = self.find_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        parameters = tool["function"]["parameters"]
        properties = parameters.get("properties", {})
        if param_name not in properties:
            return False
        del properties[param_name]
        required_list = parameters.get("required", [])
        if param_name in required_list:
            required_list.remove(param_name)
        return True


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


def _prompt_yes_no(prompt: str) -> bool:
    while True:
        value = input(f"{prompt} [y/n]: ").strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _load_tools(path: str) -> ToolStore:
    if not os.path.exists(path):
        print(f"No existing file found at {path}. Starting fresh.")
        return ToolStore()
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()
    if not content:
        return ToolStore()
    if path.endswith(".jsonl"):
        tools = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        tools = json.loads(content)
    return ToolStore(tools=tools)


def _save_tools(path: str, tools: List[Dict[str, Dict[str, object]]], file_format: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if file_format == "jsonl":
        with open(path, "w", encoding="utf-8") as file:
            for tool in tools:
                file.write(json.dumps(tool, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(tools, file, ensure_ascii=False, indent=2)
            file.write("\n")


def _select_tool(store: ToolStore) -> str:
    if not store.tools:
        raise ValueError("No tools defined yet.")
    tool_names = [tool["function"]["name"] for tool in store.tools]
    return _prompt_choice("Select a tool:", tool_names)


def _print_tools(store: ToolStore) -> None:
    if not store.tools:
        print("No tools defined yet.")
        return
    print("\nCurrent tools:")
    rows = []
    for tool in store.tools:
        fn = tool["function"]
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))
        if not properties:
            rows.append([name, desc, "-", "-", "-", "-"])
            continue
        for param_name, schema in properties.items():
            ptype = schema.get("type", "STRING")
            pdesc = schema.get("description", "")
            preq = "yes" if param_name in required else "no"
            rows.append([name, desc, param_name, ptype, preq, pdesc])

    headers = ["Tool", "Description", "Param", "Type", "Required", "Param Description"]
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


def main() -> None:
    print("FunctionGemma Tool Builder")
    file_format = _prompt_choice("Choose output format:", ["jsonl", "json"])
    default_path = f"tools.{file_format}"
    path = input(f"Enter path to save/load [{default_path}]: ").strip() or default_path

    store = _load_tools(path)

    while True:
        action = _prompt_choice(
            "\nWhat would you like to do?",
            [
                "Add a new tool/function",
                "Add a parameter to an existing tool",
                "Delete a parameter from an existing tool",
                "Delete a tool/function",
                "View current tools",
                "Save and exit",
                "Exit without saving",
            ],
        )

        if action == "Add a new tool/function":
            name = _prompt_non_empty("Function name: ")
            if store.find_tool(name):
                print("Tool already exists.")
                continue
            description = _prompt_non_empty("Function description: ")
            store.add_tool(name, description)
            print(f"Added tool '{name}'.")
        elif action == "Add a parameter to an existing tool":
            try:
                tool_name = _select_tool(store)
            except ValueError as exc:
                print(exc)
                continue
            param_name = _prompt_non_empty("Parameter name: ")
            param_type = _prompt_choice("Parameter type:", PARAMETER_TYPES)
            description = input("Parameter description (optional): ").strip()
            array_item_type = None
            if param_type == "ARRAY":
                array_item_type = _prompt_choice("Array item type:", PARAMETER_TYPES)
            required = _prompt_yes_no("Is this parameter required?")
            store.add_parameter(
                tool_name=tool_name,
                param_name=param_name,
                param_type=param_type,
                description=description,
                required=required,
                array_item_type=array_item_type,
            )
            print(f"Added parameter '{param_name}' to '{tool_name}'.")
        elif action == "Delete a parameter from an existing tool":
            try:
                tool_name = _select_tool(store)
            except ValueError as exc:
                print(exc)
                continue
            tool = store.find_tool(tool_name)
            properties = tool["function"]["parameters"].get("properties", {})
            if not properties:
                print("Selected tool has no parameters.")
                continue
            param_name = _prompt_choice("Select a parameter to delete:", list(properties.keys()))
            if store.delete_parameter(tool_name, param_name):
                print(f"Deleted parameter '{param_name}' from '{tool_name}'.")
            else:
                print("Parameter not found.")
        elif action == "Delete a tool/function":
            try:
                tool_name = _select_tool(store)
            except ValueError as exc:
                print(exc)
                continue
            if store.delete_tool(tool_name):
                print(f"Deleted tool '{tool_name}'.")
            else:
                print("Tool not found.")
        elif action == "View current tools":
            _print_tools(store)
        elif action == "Save and exit":
            _save_tools(path, store.tools, file_format=file_format)
            print(f"Saved tools to {path}.")
            break
        elif action == "Exit without saving":
            print("Exiting without saving.")
            break


if __name__ == "__main__":
    main()
