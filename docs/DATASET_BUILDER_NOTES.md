# dataset_builder.py Notes

This document explains how to use `dataset_builder.py`, including its CLI flags and interactive menu.

## Purpose
`dataset_builder.py` creates and edits FunctionGemma datasets in JSON or JSONL format. Each record includes:

```json
{
  "messages": [...],
  "tools": [...],
  "metadata": "train"
}
```

## Interactive Mode
Run without CLI flags to open the menu:

```bash
python dataset_builder.py
```

Main menu options include:
1. Add a new example
2. Add a new example for last used tool
3. View dataset size
4. Show dataset for a tool
5. Set instruction role/prompt
6. Delete entry for a tool
7. Save and exit
8. Exit without saving

## CLI Mode
When any CLI add flags are used, the menu is skipped and one record is appended.

### Core Flags
- `--dataset`  
  Path to the dataset file (JSON/JSONL).  
  Default: `dataset.jsonl`

- `--tools`  
  Path to tools file. If omitted, tools are extracted from the dataset.

### Add‑Single‑Record Flags
- `--tool` / `--tool-name`  
  Name of the tool to use for the example.

- `--prompt`  
  User prompt to insert into the example.

- `--metadata`  
  Split label for the record.  
  Default: `train`

- `--arg key=value`  
  Tool argument in `key=value` form. Can be repeated.

- `--arg-json`  
  Tool arguments as a JSON object string.  
  Merged with `--arg` (keys in `--arg` override).

- `--only-tool`  
  Include only the selected tool in the `tools` list for that record.  
  By default, all tools are included.

- `--instruction-role`  
  Role for the instruction message (`developer`, `system`, or `user`).  
  Default: last used in dataset, falling back to `developer`.

- `--instruction-prompt`  
  Instruction message content.  
  Default: last used in dataset, falling back to the template with current date/time.

### Examples

```bash
python dataset_builder.py \
  --dataset data/mud.jsonl \
  --tools data/mud_tools.json \
  --tool use \
  --prompt "Use potion of healing on me" \
  --arg object="potion of healing" \
  --arg target="me"
```

Using JSON for arguments:

```bash
python dataset_builder.py \
  --dataset data/mud.jsonl \
  --tool use \
  --prompt "Use wand of fire on the troll" \
  --arg-json '{"object":"wand of fire","target":"troll"}'
```

## Tips
1. Ensure tool calls match the tool schema (required arguments must be present).
2. Keep `metadata` limited to `train` or `eval` so `train.py` splits correctly.
3. If you set a custom instruction role/prompt, keep it consistent across train and eval.
