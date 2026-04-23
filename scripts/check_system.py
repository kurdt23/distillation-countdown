from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def try_nvidia_smi() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "path": None, "output": None}

    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {"available": True, "path": executable, "output": lines}
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"available": True, "path": executable, "output": [f"nvidia-smi failed: {type(exc).__name__}: {exc}"]}


def main() -> None:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi": try_nvidia_smi(),
    }

    if torch.cuda.is_available():
        report["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gb": round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 2),
            }
            for index in range(torch.cuda.device_count())
        ]
    else:
        report["cuda_devices"] = []

    repo_root = Path(__file__).resolve().parents[1]
    report["recommended_train_command_windows"] = (
        f"python {repo_root / 'scripts' / 'train_sft.py'} "
        f"--train-jsonl {repo_root / 'data' / 'processed' / 'highscore' / 'train.jsonl'} "
        f"--val-jsonl {repo_root / 'data' / 'processed' / 'highscore' / 'val.jsonl'} "
        f"--output-dir {repo_root / 'outputs' / 'gemma-highscore-v1'} "
        "--per-device-train-batch-size 2 --per-device-eval-batch-size 2 "
        "--gradient-accumulation-steps 16 --num-train-epochs 2.0 "
        "--learning-rate 1.5e-4 --weight-decay 0.01 --warmup-ratio 0.03 "
        "--max-seq-length 192 --gradient-checkpointing "
        "--lora-r 32 --lora-alpha 64 --lora-dropout 0.05 "
        "--logging-steps 25 --eval-steps 250 --save-steps 250 --save-total-limit 3"
    )
    report["recommended_train_command_wsl"] = (
        f"python {repo_root / 'scripts' / 'train_sft.py'} "
        f"--train-jsonl {repo_root / 'data' / 'processed' / 'highscore' / 'train.jsonl'} "
        f"--val-jsonl {repo_root / 'data' / 'processed' / 'highscore' / 'val.jsonl'} "
        f"--output-dir {repo_root / 'outputs' / 'gemma-highscore-v1'} "
        "--load-in-4bit --per-device-train-batch-size 4 --per-device-eval-batch-size 4 "
        "--gradient-accumulation-steps 8 --num-train-epochs 2.5 "
        "--learning-rate 1.5e-4 --weight-decay 0.01 --warmup-ratio 0.03 "
        "--max-seq-length 192 --gradient-checkpointing "
        "--lora-r 32 --lora-alpha 64 --lora-dropout 0.05 "
        "--logging-steps 25 --eval-steps 250 --save-steps 250 --save-total-limit 3"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

