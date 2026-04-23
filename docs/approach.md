# Approach

## Goal

Train `google/gemma-3-1b-it` for Countdown-style arithmetic generation with a reproducible local pipeline:

- dataset preparation;
- LoRA-SFT training;
- inference and submission export;
- local validity checks.

## Data sources

The project supports three label sources:

1. Verified teacher traces from the public dataset:
   - `verified_Qwen2.5-7B-Instruct`
   - `verified_Qwen3-4B-Instruct-2507`
2. Exact solver labels on public generator outputs:
   - `all:train`
3. Optional additional teacher generations from an API or local model.

All labels are normalized through the same expression parser and validator.

## Main scripts

- `scripts/prepare_dataset.py`: builds `train.jsonl` and `val.jsonl`
- `scripts/train_sft.py`: LoRA-SFT for `google/gemma-3-1b-it`
- `scripts/predict.py`: inference and submission export
- `scripts/evaluate_predictions.py`: local accuracy check
- `scripts/generate_teacher.py`: optional teacher generation
- `scripts/merge_adapter.py`: adapter merge

## Inference policy

Inference remains model-driven:

- multiple candidates can be generated per prompt;
- candidates are filtered with the local validator;
- optional repair asks the same model to correct an invalid expression.

The project does not apply an exact solver to the competition submission file at inference time.

## Runtime notes

- Native Windows is supported for data preparation and CPU-safe runs.
- GPU training is more reliable in WSL2 or Linux, especially for `4-bit` loading.
- Use `scripts/check_system.py` before training to verify that CUDA is visible to PyTorch.
