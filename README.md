# Countdown Distillation

Локальный проект для обучения `google/gemma-3-1b-it` Countdown-задачи.

## Структура

- `countdown_distill/` — парсер выражений, solver, prompt helpers, загрузка данных
- `scripts/prepare_dataset.py` — сборка train/val датасета
- `scripts/train_sft.py` — LoRA-SFT
- `scripts/predict.py` — инференс и `submission.csv`
- `scripts/evaluate_predictions.py` — локальная проверка accuracy
- `scripts/generate_teacher.py` — опциональная teacher-генерация
- `scripts/check_system.py` — проверка CUDA и окружения
- `docs/approach.md` — краткие технические заметки
- `notebooks/colab-a100.ipynb` — ноутбук обучения `Qwen2.5-7B` в колабе
- `scripts/merge_adapter.py` — объединение с первой стадией для дообучения второй
- `scripts/ensemble_predictions.py` — ансамблирование предикта

Проект сделан под выполнение в среде WSL2 / Linux.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_system.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Данные

Генерация примеров учителем:

```bash
python scripts/generate_teacher.py \
  --backend transformers --model-id Qwen/Qwen3-8B \
  --dataset-config all --split train --shuffle --seed 42 \
  --limit 50000 --batch-size 32 --num-candidates 8 \
  --temperature 0.6 --top-p 0.95 --max-new-tokens 96 \
  --load-in-4bit --repair-with-solver \
  --output-jsonl data/generated/qwen3_8b_train50k_8cand.jsonl
```

## Подготовка:

```bash
python scripts/prepare_dataset.py \
  --output-dir data/processed/strong50_stage1 \
  --dedupe-key task_label \
  --verified-config verified_Qwen2.5-7B-Instruct \
  --verified-config verified_Qwen3-4B-Instruct-2507 \
  --verified-config verified_Qwen2.5-0.5B-Instruct \
  --extra-jsonl data/generated/qwen3_8b_train50k_8cand.jsonl \
  --canonicalize-with-solver --repair-with-solver \
  --solver-source all:train --val-size 2000 --seed 42
```
 
## Обучение

Stage 1:

```bash
python scripts/train_sft.py \
  --train-jsonl data/processed/strong50_stage1/train.jsonl \
  --val-jsonl data/processed/strong50_stage1/val.jsonl \
  --output-dir outputs/strong50-stage1 \
  --model-id google/gemma-3-1b-it --load-in-4bit \
  --per-device-train-batch-size 16 --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 2 --num-train-epochs 6.0 \
  --learning-rate 1.5e-4 --weight-decay 0.01 --warmup-ratio 0.03 \
  --max-seq-length 192 --gradient-checkpointing \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05
```

Stage 2 (подготовка+слияние):

- подготовка:

```bash
python scripts/prepare_dataset.py \
  --output-dir data/processed/strong50_stage2 \
  --dedupe-key task_label \
  --verified-config verified_Qwen2.5-7B-Instruct \
  --verified-config verified_Qwen3-4B-Instruct-2507 \
  --verified-config verified_Qwen2.5-0.5B-Instruct \
  --extra-jsonl data/generated/qwen3_8b_train50k_8cand.jsonl \
  --canonicalize-with-solver --repair-with-solver \
  --val-size 2000 --seed 42
```

- слияние адаптера со Stage 1:

```bash
python scripts/merge_adapter.py \
  --adapter-dir outputs/strong50-stage1/adapter \
  --base-model-id google/gemma-3-1b-it \
  --output-dir outputs/strong50-stage1-merged
```

- обучение:

```bash
python scripts/train_sft.py \
  --train-jsonl data/processed/strong50_stage2/train.jsonl \
  --val-jsonl data/processed/strong50_stage2/val.jsonl \
  --output-dir outputs/strong50-stage2 \
  --model-id outputs/strong50-stage1-merged --load-in-4bit \
  --per-device-train-batch-size 16 --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --num-train-epochs 3.0 --learning-rate 5e-5 \
  --weight-decay 0.01 --warmup-ratio 0.03 --max-seq-length 192 \
  --gradient-checkpointing --lora-r 32 --lora-alpha 64 --lora-dropout 0.05
```


## Инференс

Stage 1:

```bash
python scripts/predict.py \
  --model-path outputs/strong50-stage1/adapter \
  --base-model-id google/gemma-3-1b-it \
  --input-path data/test_public.csv \
  --output-csv outputs/submission_stage1.csv \
  --load-in-4bit --batch-size 2 --num-candidates 128 \
  --temperature 0.8 --top-p 0.98 --max-new-tokens 192 \
  --repair-attempts 2 --repair-num-candidates 128 \
  --repair-temperature 0.5 --repair-top-p 0.96 \
  --keep-invalid --restrict-output-charset --progress-every 1
```

Stage 2:

```bash
python scripts/predict.py \
  --model-path outputs/strong50-stage2/adapter \
  --base-model-id google/gemma-3-1b-it \
  --input-path data/test_public.csv \
  --output-csv outputs/submission_stage2.csv \
  --load-in-4bit --batch-size 2 --num-candidates 128 \
  --temperature 0.8 --top-p 0.98 --max-new-tokens 192 \
  --repair-attempts 2 --repair-num-candidates 128 \
  --repair-temperature 0.5 --repair-top-p 0.96 \
  --keep-invalid --restrict-output-charset --progress-every 1
```

## Ансамбль двухэтапного локального обучения + обучения в колабе

```bash
python scripts/ensemble_predictions.py \
  --reference data/test_public.csv \
  --input-csv outputs/submission_stage1.csv \
  --input-csv outputs/submission_stage2.csv \
  --input-csv outputs/submission_colab.csv \
  --output-csv outputs/submission_ensemble.csv
```

## Локальная оценка

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/submission_ensemble.csv \
  --reference data/test_public.csv
```

