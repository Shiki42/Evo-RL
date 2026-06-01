#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from pathlib import Path

import torch

from scripts.rlt_online_cycle import (
    DEFAULT_BASE_DIR,
    DEFAULT_BASE_RLT_POLICY_DIR,
    DEFAULT_MODEL_REPO,
    DEFAULT_TASK,
    build_transition_cache,
    emit_result,
    sanitize_name,
    snapshot_dataset,
    snapshot_model_base,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RLT online transition cache")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--base-rlt-policy-dir", default=DEFAULT_BASE_RLT_POLICY_DIR)
    parser.add_argument("--cache-root", default="/home/coder/share/cache/rlt_online")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root).expanduser().resolve()
    run_name = args.run_name or sanitize_name(args.dataset_repo_id)
    transition_cache = cache_root / "transition_cache" / run_name
    transition_cache.mkdir(parents=True, exist_ok=True)
    train_cache = transition_cache / "chunk_transitions_train.pt"
    val_cache = transition_cache / "chunk_transitions_val.pt"
    if args.rebuild or not (train_cache.is_file() and val_cache.is_file()):
        model_snapshot = snapshot_model_base(args.model_repo, args.base_dir, cache_root)
        dataset_root = snapshot_dataset(args.dataset_repo_id, cache_root)
        build_transition_cache(args, dataset_root, model_snapshot, transition_cache)
    train = torch.load(train_cache, map_location="cpu", weights_only=False)
    val = torch.load(val_cache, map_location="cpu", weights_only=False)
    emit_result({
        "dataset_repo_id": args.dataset_repo_id,
        "transition_cache_dir": str(transition_cache),
        "train_transitions": len(train),
        "val_transitions": len(val),
    })


if __name__ == "__main__":
    main()
