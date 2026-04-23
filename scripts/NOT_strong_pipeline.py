from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERIFIED_CONFIGS = [
    "verified_Qwen2.5-7B-Instruct",
    "verified_Qwen3-4B-Instruct-2507",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the strongest competition-safe two-stage Countdown pipeline on a CUDA machine."
    )
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=[
            "teacher",
            "prepare-stage1",
            "train-stage1",
            "merge-stage1",
            "prepare-stage2",
            "train-stage2",
            "onpolicy",
            "train-onpolicy-sft",
            "train-dpo",
            "predict",
            "eval",
            "all-train",
            "all-onpolicy",
            "all",
        ],
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for subprocesses.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verified-config", dest="verified_configs", action="append", default=[])
    parser.add_argument("--stage1-dedupe-key", choices=["task", "task_label"], default="task_label")
    parser.add_argument("--stage2-dedupe-key", choices=["task", "task_label"], default="task_label")

    parser.add_argument("--teacher-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--teacher-output", type=Path, default=REPO_ROOT / "data" / "generated" / "qwen3_8b_train20k.jsonl")
    parser.add_argument("--teacher-limit", type=int, default=20000)
    parser.add_argument("--teacher-batch-size", type=int, default=2)
    parser.add_argument("--teacher-num-candidates", type=int, default=4)
    parser.add_argument("--teacher-temperature", type=float, default=0.6)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=96)
    parser.add_argument("--teacher-progress-every", type=int, default=100)

    parser.add_argument("--stage1-data-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "strong_stage1")
    parser.add_argument("--stage2-data-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "strong_stage2")
    parser.add_argument("--val-size", type=int, default=2000)

    parser.add_argument("--student-model", default="google/gemma-3-1b-it")
    parser.add_argument("--stage1-output-dir", type=Path, default=REPO_ROOT / "outputs" / "strong-stage1")
    parser.add_argument("--stage1-merged-dir", type=Path, default=REPO_ROOT / "outputs" / "strong-stage1-merged")
    parser.add_argument("--stage2-output-dir", type=Path, default=REPO_ROOT / "outputs" / "strong-stage2")

    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=192)
    parser.add_argument("--stage1-epochs", type=float, default=1.5)
    parser.add_argument("--stage2-epochs", type=float, default=1.0)
    parser.add_argument("--stage1-learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--stage1-eval-steps", type=int, default=500)
    parser.add_argument("--stage1-save-steps", type=int, default=500)
    parser.add_argument("--stage2-eval-steps", type=int, default=250)
    parser.add_argument("--stage2-save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--predict-input", type=Path, default=REPO_ROOT / "data" / "test_public.csv")
    parser.add_argument("--predict-output", type=Path, default=REPO_ROOT / "outputs" / "submission_strong_stage2.csv")
    parser.add_argument("--predict-debug-jsonl", type=Path, default=REPO_ROOT / "outputs" / "predict_strong_stage2.jsonl")
    parser.add_argument("--predict-source", choices=["stage1", "stage2", "onpolicy-sft", "dpo"], default="stage2")
    parser.add_argument("--predict-batch-size", type=int, default=4)
    parser.add_argument("--predict-num-candidates", type=int, default=8)
    parser.add_argument("--predict-temperature", type=float, default=0.7)
    parser.add_argument("--predict-top-p", type=float, default=0.95)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--repair-num-candidates", type=int, default=4)
    parser.add_argument("--repair-temperature", type=float, default=0.35)
    parser.add_argument("--repair-top-p", type=float, default=0.90)
    parser.add_argument("--predict-progress-every", type=int, default=10)
    parser.add_argument("--onpolicy-source-jsonl", type=Path, default=None)
    parser.add_argument("--onpolicy-output-dir", type=Path, default=REPO_ROOT / "data" / "on_policy" / "round1")
    parser.add_argument("--onpolicy-model-source", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--onpolicy-limit", type=int, default=30000)
    parser.add_argument("--onpolicy-val-size", type=int, default=1000)
    parser.add_argument("--onpolicy-batch-size", type=int, default=4)
    parser.add_argument("--onpolicy-num-candidates", type=int, default=32)
    parser.add_argument("--onpolicy-temperature", type=float, default=0.8)
    parser.add_argument("--onpolicy-top-p", type=float, default=0.95)
    parser.add_argument("--onpolicy-max-new-tokens", type=int, default=64)
    parser.add_argument("--onpolicy-max-dpo-pairs-per-task", type=int, default=2)
    parser.add_argument("--onpolicy-sft-output-dir", type=Path, default=REPO_ROOT / "outputs" / "onpolicy-sft-r1")
    parser.add_argument("--onpolicy-sft-epochs", type=float, default=1.0)
    parser.add_argument("--onpolicy-sft-learning-rate", type=float, default=5e-5)
    parser.add_argument("--dpo-output-dir", type=Path, default=REPO_ROOT / "outputs" / "dpo-r1")
    parser.add_argument("--dpo-epochs", type=float, default=1.0)
    parser.add_argument("--dpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.05)
    parser.add_argument("--dpo-max-length", type=int, default=256)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=192)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_verified_configs(args: argparse.Namespace) -> list[str]:
    return args.verified_configs or list(DEFAULT_VERIFIED_CONFIGS)


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def ensure_exists_unless_dry_run(path: Path, description: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    ensure_exists(path, description)


def quoted_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, dry_run: bool) -> None:
    print()
    print(f"$ {quoted_command(command)}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def build_teacher_command(args: argparse.Namespace) -> list[str]:
    args.teacher_output.parent.mkdir(parents=True, exist_ok=True)
    return [
        args.python,
        str(REPO_ROOT / "scripts" / "generate_teacher.py"),
        "--backend",
        "transformers",
        "--model-id",
        args.teacher_model,
        "--dataset-config",
        "all",
        "--split",
        "train",
        "--shuffle",
        "--seed",
        str(args.seed),
        "--limit",
        str(args.teacher_limit),
        "--batch-size",
        str(args.teacher_batch_size),
        "--num-candidates",
        str(args.teacher_num_candidates),
        "--temperature",
        str(args.teacher_temperature),
        "--top-p",
        str(args.teacher_top_p),
        "--max-new-tokens",
        str(args.teacher_max_new_tokens),
        "--progress-every",
        str(args.teacher_progress_every),
        "--load-in-4bit",
        "--repair-with-solver",
        "--output-jsonl",
        str(args.teacher_output),
    ]


def build_prepare_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    include_solver_source: bool,
    dedupe_key: str,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "prepare_dataset.py"),
        "--output-dir",
        str(output_dir),
        "--dedupe-key",
        dedupe_key,
    ]
    for config_name in resolved_verified_configs(args):
        command.extend(["--verified-config", config_name])
    command.extend(
        [
            "--extra-jsonl",
            str(args.teacher_output),
            "--canonicalize-with-solver",
            "--repair-with-solver",
            "--val-size",
            str(args.val_size),
            "--seed",
            str(args.seed),
        ]
    )
    if include_solver_source:
        command.extend(["--solver-source", "all:train"])
    return command


def build_train_command(
    args: argparse.Namespace,
    *,
    train_jsonl: Path,
    val_jsonl: Path,
    output_dir: Path,
    model_id: str | Path,
    epochs: float,
    learning_rate: float,
    eval_steps: int,
    save_steps: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        args.python,
        str(REPO_ROOT / "scripts" / "train_sft.py"),
        "--train-jsonl",
        str(train_jsonl),
        "--val-jsonl",
        str(val_jsonl),
        "--output-dir",
        str(output_dir),
        "--model-id",
        str(model_id),
        "--base-model-id",
        args.student_model,
        "--load-in-4bit",
        "--per-device-train-batch-size",
        str(args.train_batch_size),
        "--per-device-eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--num-train-epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-seq-length",
        str(args.max_seq_length),
        "--gradient-checkpointing",
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(eval_steps),
        "--save-steps",
        str(save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--seed",
        str(args.seed),
    ]


def build_merge_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(REPO_ROOT / "scripts" / "merge_adapter.py"),
        "--adapter-dir",
        str(args.stage1_output_dir / "adapter"),
        "--base-model-id",
        args.student_model,
        "--output-dir",
        str(args.stage1_merged_dir),
    ]


def build_predict_command(args: argparse.Namespace) -> list[str]:
    args.predict_output.parent.mkdir(parents=True, exist_ok=True)
    args.predict_debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
    model_path = model_path_for_source(args, args.predict_source)
    return [
        args.python,
        str(REPO_ROOT / "scripts" / "predict.py"),
        "--model-path",
        str(model_path),
        "--base-model-id",
        args.student_model,
        "--input-path",
        str(args.predict_input),
        "--output-csv",
        str(args.predict_output),
        "--load-in-4bit",
        "--batch-size",
        str(args.predict_batch_size),
        "--num-candidates",
        str(args.predict_num_candidates),
        "--temperature",
        str(args.predict_temperature),
        "--top-p",
        str(args.predict_top_p),
        "--repair-attempts",
        str(args.repair_attempts),
        "--repair-num-candidates",
        str(args.repair_num_candidates),
        "--repair-temperature",
        str(args.repair_temperature),
        "--repair-top-p",
        str(args.repair_top_p),
        "--progress-every",
        str(args.predict_progress_every),
        "--debug-jsonl",
        str(args.predict_debug_jsonl),
    ]


def model_path_for_source(args: argparse.Namespace, source: str) -> Path:
    mapping = {
        "stage1": args.stage1_output_dir / "adapter",
        "stage2": args.stage2_output_dir / "adapter",
        "onpolicy-sft": args.onpolicy_sft_output_dir / "adapter",
        "dpo": args.dpo_output_dir / "adapter",
    }
    return mapping[source]


def onpolicy_dir_file(args: argparse.Namespace, name: str) -> Path:
    return args.onpolicy_output_dir / name


def build_onpolicy_command(args: argparse.Namespace) -> list[str]:
    args.onpolicy_output_dir.mkdir(parents=True, exist_ok=True)
    source_jsonl = args.onpolicy_source_jsonl or args.stage1_data_dir / "train.jsonl"
    model_path = model_path_for_source(args, args.onpolicy_model_source)
    return [
        args.python,
        str(REPO_ROOT / "scripts" / "generate_on_policy.py"),
        "--model-path",
        str(model_path),
        "--base-model-id",
        args.student_model,
        "--source-jsonl",
        str(source_jsonl),
        "--output-sft-train-jsonl",
        str(onpolicy_dir_file(args, "sft_train.jsonl")),
        "--output-sft-val-jsonl",
        str(onpolicy_dir_file(args, "sft_val.jsonl")),
        "--output-dpo-train-jsonl",
        str(onpolicy_dir_file(args, "dpo_train.jsonl")),
        "--output-dpo-val-jsonl",
        str(onpolicy_dir_file(args, "dpo_val.jsonl")),
        "--stats-json",
        str(onpolicy_dir_file(args, "stats.json")),
        "--shuffle",
        "--seed",
        str(args.seed),
        "--limit",
        str(args.onpolicy_limit),
        "--val-size",
        str(args.onpolicy_val_size),
        "--batch-size",
        str(args.onpolicy_batch_size),
        "--num-candidates",
        str(args.onpolicy_num_candidates),
        "--temperature",
        str(args.onpolicy_temperature),
        "--top-p",
        str(args.onpolicy_top_p),
        "--max-new-tokens",
        str(args.onpolicy_max_new_tokens),
        "--max-dpo-pairs-per-task",
        str(args.onpolicy_max_dpo_pairs_per_task),
        "--load-in-4bit",
        "--solver-chosen-for-dpo",
        "--progress-every",
        "10",
    ]


def build_train_dpo_command(args: argparse.Namespace) -> list[str]:
    args.dpo_output_dir.mkdir(parents=True, exist_ok=True)
    return [
        args.python,
        str(REPO_ROOT / "scripts" / "train_dpo.py"),
        "--train-jsonl",
        str(onpolicy_dir_file(args, "dpo_train.jsonl")),
        "--val-jsonl",
        str(onpolicy_dir_file(args, "dpo_val.jsonl")),
        "--output-dir",
        str(args.dpo_output_dir),
        "--model-id",
        str(args.onpolicy_sft_output_dir / "adapter"),
        "--base-model-id",
        args.student_model,
        "--load-in-4bit",
        "--per-device-train-batch-size",
        str(args.train_batch_size),
        "--per-device-eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--num-train-epochs",
        str(args.dpo_epochs),
        "--learning-rate",
        str(args.dpo_learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--beta",
        str(args.dpo_beta),
        "--max-length",
        str(args.dpo_max_length),
        "--max-prompt-length",
        str(args.dpo_max_prompt_length),
        "--gradient-checkpointing",
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(args.stage2_eval_steps),
        "--save-steps",
        str(args.stage2_save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--seed",
        str(args.seed),
    ]


def build_eval_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(REPO_ROOT / "scripts" / "evaluate_predictions.py"),
        "--predictions",
        str(args.predict_output),
        "--reference",
        str(args.predict_input),
    ]


def execute_step(args: argparse.Namespace, step: str) -> None:
    if step == "teacher":
        run_command(build_teacher_command(args), dry_run=args.dry_run)
        return

    if step == "prepare-stage1":
        ensure_exists_unless_dry_run(args.teacher_output, "teacher JSONL", dry_run=args.dry_run)
        run_command(
            build_prepare_command(
                args,
                output_dir=args.stage1_data_dir,
                include_solver_source=True,
                dedupe_key=args.stage1_dedupe_key,
            ),
            dry_run=args.dry_run,
        )
        return

    if step == "train-stage1":
        ensure_exists_unless_dry_run(args.stage1_data_dir / "train.jsonl", "stage 1 train.jsonl", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(args.stage1_data_dir / "val.jsonl", "stage 1 val.jsonl", dry_run=args.dry_run)
        run_command(
            build_train_command(
                args,
                train_jsonl=args.stage1_data_dir / "train.jsonl",
                val_jsonl=args.stage1_data_dir / "val.jsonl",
                output_dir=args.stage1_output_dir,
                model_id=args.student_model,
                epochs=args.stage1_epochs,
                learning_rate=args.stage1_learning_rate,
                eval_steps=args.stage1_eval_steps,
                save_steps=args.stage1_save_steps,
            ),
            dry_run=args.dry_run,
        )
        return

    if step == "merge-stage1":
        ensure_exists_unless_dry_run(args.stage1_output_dir / "adapter", "stage 1 adapter", dry_run=args.dry_run)
        run_command(build_merge_command(args), dry_run=args.dry_run)
        return

    if step == "prepare-stage2":
        ensure_exists_unless_dry_run(args.teacher_output, "teacher JSONL", dry_run=args.dry_run)
        run_command(
            build_prepare_command(
                args,
                output_dir=args.stage2_data_dir,
                include_solver_source=False,
                dedupe_key=args.stage2_dedupe_key,
            ),
            dry_run=args.dry_run,
        )
        return

    if step == "train-stage2":
        ensure_exists_unless_dry_run(args.stage2_data_dir / "train.jsonl", "stage 2 train.jsonl", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(args.stage2_data_dir / "val.jsonl", "stage 2 val.jsonl", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(args.stage1_merged_dir, "stage 1 merged model", dry_run=args.dry_run)
        run_command(
            build_train_command(
                args,
                train_jsonl=args.stage2_data_dir / "train.jsonl",
                val_jsonl=args.stage2_data_dir / "val.jsonl",
                output_dir=args.stage2_output_dir,
                model_id=args.stage1_merged_dir,
                epochs=args.stage2_epochs,
                learning_rate=args.stage2_learning_rate,
                eval_steps=args.stage2_eval_steps,
                save_steps=args.stage2_save_steps,
            ),
            dry_run=args.dry_run,
        )
        return

    if step == "onpolicy":
        source_jsonl = args.onpolicy_source_jsonl or args.stage1_data_dir / "train.jsonl"
        selected_adapter = model_path_for_source(args, args.onpolicy_model_source)
        ensure_exists_unless_dry_run(source_jsonl, "on-policy source JSONL", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(selected_adapter, f"{args.onpolicy_model_source} adapter", dry_run=args.dry_run)
        run_command(build_onpolicy_command(args), dry_run=args.dry_run)
        return

    if step == "train-onpolicy-sft":
        ensure_exists_unless_dry_run(onpolicy_dir_file(args, "sft_train.jsonl"), "on-policy SFT train JSONL", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(onpolicy_dir_file(args, "sft_val.jsonl"), "on-policy SFT val JSONL", dry_run=args.dry_run)
        run_command(
            build_train_command(
                args,
                train_jsonl=onpolicy_dir_file(args, "sft_train.jsonl"),
                val_jsonl=onpolicy_dir_file(args, "sft_val.jsonl"),
                output_dir=args.onpolicy_sft_output_dir,
                model_id=model_path_for_source(args, args.onpolicy_model_source),
                epochs=args.onpolicy_sft_epochs,
                learning_rate=args.onpolicy_sft_learning_rate,
                eval_steps=args.stage2_eval_steps,
                save_steps=args.stage2_save_steps,
            ),
            dry_run=args.dry_run,
        )
        return

    if step == "train-dpo":
        ensure_exists_unless_dry_run(onpolicy_dir_file(args, "dpo_train.jsonl"), "DPO train JSONL", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(onpolicy_dir_file(args, "dpo_val.jsonl"), "DPO val JSONL", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(args.onpolicy_sft_output_dir / "adapter", "on-policy SFT adapter", dry_run=args.dry_run)
        run_command(build_train_dpo_command(args), dry_run=args.dry_run)
        return

    if step == "predict":
        selected_adapter = model_path_for_source(args, args.predict_source)
        ensure_exists_unless_dry_run(selected_adapter, f"{args.predict_source} adapter", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(args.predict_input, "prediction input file", dry_run=args.dry_run)
        run_command(build_predict_command(args), dry_run=args.dry_run)
        return

    if step == "eval":
        ensure_exists_unless_dry_run(args.predict_output, "prediction output CSV", dry_run=args.dry_run)
        ensure_exists_unless_dry_run(args.predict_input, "evaluation reference CSV", dry_run=args.dry_run)
        run_command(build_eval_command(args), dry_run=args.dry_run)
        return

    raise ValueError(f"Unsupported step: {step}")


def expand_steps(step: str) -> list[str]:
    if step == "all-train":
        return ["teacher", "prepare-stage1", "train-stage1", "merge-stage1", "prepare-stage2", "train-stage2"]
    if step == "all-onpolicy":
        return ["onpolicy", "train-onpolicy-sft", "train-dpo"]
    if step == "all":
        return ["teacher", "prepare-stage1", "train-stage1", "merge-stage1", "prepare-stage2", "train-stage2", "predict", "eval"]
    return [step]


def main() -> None:
    args = parse_args()
    for step in expand_steps(args.step):
        print(f"\n=== {step} ===")
        execute_step(args, step)


if __name__ == "__main__":
    main()
