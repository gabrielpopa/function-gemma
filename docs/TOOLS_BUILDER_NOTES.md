# tools_builder.py Notes

This document explains how to use `tools_builder.py`, what it creates, and how it fits into dataset creation.

## Purpose
`tools_builder.py` is an interactive CLI for creating and editing tool/function schemas. It produces a JSON or JSONL file containing function definitions compatible with FunctionGemma datasets.

## Output Format
Each tool is stored as:

```json
{
  "function": {
    "name": "tool_name",
    "description": "What this tool does.",
    "parameters": {
      "type": "OBJECT",
      "properties": {
        "param": { "type": "STRING", "description": "..." }
      },
      "required": ["param"]
    }
  }
}
```

## Features
1. Create new tools/functions with names and descriptions.
2. Add parameters with types (`STRING`, `INTEGER`, `NUMBER`, `BOOLEAN`, `ARRAY`, `OBJECT`).
3. Mark parameters as required or optional.
4. Delete a parameter from an existing tool.
5. Delete an entire tool.
6. View tools in a readable table format.
7. Load and save in JSON or JSONL (default is JSONL).

## Usage

```bash
python tools_builder.py
```

You will be prompted to:
1. Choose output format (defaults to JSONL).
2. Load an existing tools file or create a new one.
3. Add/edit/delete tools and parameters.
4. Save changes to disk.

## Tips
1. Use **JSONL** when you want to append tools easily.
2. Keep tool names stable once used in datasets (changing names invalidates old examples).
3. Required parameters should match what you expect in real tool calls.

## Related
- Use `dataset_builder.py` to build training examples from these tools.
- Use `ollama_dataset_generator.py` to auto‑generate examples for a tool.
