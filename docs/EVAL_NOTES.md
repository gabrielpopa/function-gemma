# Train vs Eval Notes (Mobile Actions)

The Mobile Actions dataset is stored as JSONL and uses a `metadata` field to mark the split for each record. The dataset card specifies that `metadata` is either `train` or `eval`, and the split is pre-defined by that field. citeturn2view0

## Is `eval` needed?
Strictly speaking, you can train without an eval split, but you lose a reliable way to measure quality, detect overfitting, and compare model changes. For any dataset you care about improving, an eval split is strongly recommended.

## How is `eval` different from `train`?
`train` examples are used to update model weights during fine‑tuning.  
`eval` examples are held out and used only for validation/measurement. They must not be used for training or hyperparameter fitting beyond selecting the best checkpoint.

## Can `eval` be the same as `train`?
No. If you evaluate on the same data you trained on, you measure memorization, not generalization.  
You *can* create an eval split by taking a subset of the same dataset, but it must be **disjoint** from train. Do not include near‑duplicates (same prompt with minor rewording or same arguments in different phrasing) in both splits.

## Practical guidance
1. Treat `eval` as held‑out data. Do not train on it.
2. Keep `tools` schemas consistent between train and eval so evaluation is meaningful.
3. When creating new data, reserve a portion for `eval` that does not overlap with train prompts or argument combinations.
4. Use `eval` for validation and comparison across model runs, not for fitting.

## Recommended eval size (rules of thumb)
These are heuristics; use what is realistic for your dataset size:
1. **Overall split**: 5–20% of the data for eval. Smaller datasets usually need a larger percentage to get a stable signal.
2. **Per‑tool coverage**: ensure *every tool* appears in eval.  
   - A simple target is **~10% per tool**, with a **minimum of 5–20 examples per tool** if possible.  
   - If a tool has very few examples, consider manual curation and keep at least a few in eval.
3. **Stratify** by tool and by argument patterns (required vs optional fields, different targets, etc.) so eval tests the same capabilities you train.

## When data is small
If you can’t afford a dedicated eval split, consider:
1. A tiny eval set (even 1–2 examples per tool) just to sanity‑check outputs.
2. K‑fold cross‑validation or repeated hold‑out splits to reduce variance.
