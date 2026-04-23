from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import HfFolder


def read_row_count(path: Path) -> int | None:
    try:
        if path.suffix.lower() == ".csv":
            return int(pd.read_csv(path).shape[0])
        if path.suffix.lower() == ".parquet":
            return int(pd.read_parquet(path).shape[0])
    except Exception:
        return None
    return None


def model_cache_exists(base_model_id: str) -> bool:
    cache_dir_name = "models--" + base_model_id.replace("/", "--")
    cache_roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".huggingface" / "hub",
    ]
    for root in cache_roots:
        snapshots_dir = root / cache_dir_name / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            return True
    return False


def check_hf_access(base_model_id: str) -> dict[str, Any]:
    token = HfFolder.get_token()
    cached = model_cache_exists(base_model_id)
    result: dict[str, Any] = {
        "token_present": bool(token),
        "cached_locally": cached,
        "remote_access_ok": False,
        "whoami": None,
        "error": None,
    }
    if not token:
        if cached:
            result["error"] = "No Hugging Face token, but the base model appears to be cached locally."
        else:
            result["error"] = "No Hugging Face token and no local cache for the base model."
        return result

    api = HfApi(token=token)
    try:
        whoami = api.whoami()
        result["whoami"] = whoami.get("name") or whoami.get("fullname") or "authenticated"
    except Exception as exc:
        result["error"] = f"whoami failed: {type(exc).__name__}: {exc}"

    try:
        api.model_info(base_model_id)
        result["remote_access_ok"] = True
    except HfHubHTTPError as exc:
        result["error"] = f"model access failed: {exc.response.status_code} {exc}"
    except Exception as exc:
        result["error"] = f"model access failed: {type(exc).__name__}: {exc}"
    return result


def print_check(ok: bool, label: str, detail: str) -> None:
    prefix = "[OK]" if ok else "[WARN]"
    print(f"{prefix} {label}: {detail}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    adapter_dir = repo_root / "outputs" / "gemma-highscore-v1" / "adapter"
    adapter_config_path = adapter_dir / "adapter_config.json"
    train_jsonl = repo_root / "data" / "processed" / "highscore_safe" / "train.jsonl"
    val_jsonl = repo_root / "data" / "processed" / "highscore_safe" / "val.jsonl"
    kaggle_input = repo_root / "data" / "test_public.csv"
    parquet_input = repo_root / "data" / "test.parquet"

    base_model_id = "google/gemma-3-1b-it"
    if adapter_config_path.exists():
        try:
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            base_model_id = adapter_config.get("base_model_name_or_path", base_model_id)
        except Exception:
            pass

    adapter_ok = adapter_dir.exists()
    safe_data_ok = train_jsonl.exists() and val_jsonl.exists()
    kaggle_input_ok = kaggle_input.exists()
    parquet_input_ok = parquet_input.exists()
    hf_status = check_hf_access(base_model_id)
    model_access_ok = bool(hf_status["cached_locally"] or hf_status["remote_access_ok"])
    cuda_ok = torch.cuda.is_available()

    kaggle_rows = read_row_count(kaggle_input) if kaggle_input_ok else None
    parquet_rows = read_row_count(parquet_input) if parquet_input_ok else None

    print_check(adapter_ok, "Adapter", str(adapter_dir))
    print_check(safe_data_ok, "Safe training dataset", f"train={train_jsonl.exists()} val={val_jsonl.exists()}")
    print_check(kaggle_input_ok, "Kaggle public test", f"{kaggle_input} rows={kaggle_rows}")
    print_check(parquet_input_ok, "Local parquet test", f"{parquet_input} rows={parquet_rows}")
    print_check(cuda_ok, "CUDA visible to PyTorch", f"device_count={torch.cuda.device_count()}")
    print_check(
        model_access_ok,
        "Gemma base model access",
        (
            f"model={base_model_id}; token_present={hf_status['token_present']}; "
            f"cached={hf_status['cached_locally']}; remote_access={hf_status['remote_access_ok']}; "
            f"detail={hf_status['error'] or hf_status['whoami'] or 'ok'}"
        ),
    )

    kaggle_predict_ready = adapter_ok and kaggle_input_ok and model_access_ok
    local_predict_ready = adapter_ok and parquet_input_ok and model_access_ok
    train_ready = safe_data_ok and cuda_ok and model_access_ok

    print()
    print("Summary:")
    print(f"- Kaggle inference ready: {'yes' if kaggle_predict_ready else 'no'}")
    print(f"- Local inference ready: {'yes' if local_predict_ready else 'no'}")
    print(f"- Training ready: {'yes' if train_ready else 'no'}")


if __name__ == "__main__":
    main()
