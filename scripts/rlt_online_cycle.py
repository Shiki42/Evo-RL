#!/usr/bin/env python
"""One-command RLT pseudo-online learning cycle.

The first reliable online-learning loop is deliberately conservative:

1. merge freshly recorded LeRobot datasets from the robot machine,
2. upload the merged dataset as a private Hugging Face dataset,
3. train actor-critic from that dataset on coder b using the newest AC ckpt,
4. upload the new raw ``rl_checkpoint.pt`` under ``ac_checkpoint/<timestamp>``,
5. download the newest AC ckpt back to the robot machine and update a symlink.

This script does not use the legacy online collectors. They remain unverified
signals; the only training path here is the already-understood chunk-transition
offline update applied repeatedly to newly collected data.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DEFAULT_MODEL_REPO = "Shiki42/pi05_screw_c_mix_cont15k_fp16"
DEFAULT_BASE_DIR = "online_base_vla_0528.pt"
DEFAULT_BASE_AC_FILE = f"{DEFAULT_BASE_DIR}/online_base_ac_0528.pt"
DEFAULT_BASE_RLT_POLICY_DIR = f"{DEFAULT_BASE_DIR}/online_base_rlt_0528.pt"
DEFAULT_TASK = "Insert the copper screw into the black sleeve."
DEFAULT_ROBOT_REPO = "/home/kye/evo-rl"
DEFAULT_CODER_B_REPO = "/home/coder/code/Evo-RL-quick"
DEFAULT_ROBOT_DATASETS_ROOT = "~/.roboclaw/workspace/embodied/datasets/local"
RESULT_PREFIX = "RLT_ONLINE_RESULT "


@dataclass(frozen=True)
class ACRef:
    version: str
    hf_path: str


@dataclass(frozen=True)
class PackResult:
    dataset_repo_id: str
    dataset_root: str
    merged_name: str
    ac_version: str
    total_episodes: int
    total_frames: int


@dataclass(frozen=True)
class TrainResult:
    ac_version: str
    checkpoint_hf_path: str
    checkpoint_local_path: str
    output_dir: str
    transition_cache_dir: str


@dataclass(frozen=True)
class DeployResult:
    ac_version: str
    checkpoint_hf_path: str
    checkpoint_local_path: str
    latest_symlink: str


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    out = []
    last_was_sep = False
    for char in value:
        if char.isalnum() or char in {".", "-"}:
            out.append(char)
            last_was_sep = False
            continue
        if not last_was_sep:
            out.append("_")
            last_was_sep = True
    return "".join(out).strip("._-") or "rlt_online"


def is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def load_dataset_info(path: Path) -> dict:
    return json.loads((path / "meta" / "info.json").read_text())


def discover_dataset_dirs(root: Path, prefixes: list[str], max_datasets: int | None) -> list[Path]:
    info_paths = sorted(root.expanduser().rglob("meta/info.json"))
    datasets = [path.parents[1] for path in info_paths]
    matches = [path for path in datasets if any(path.name.startswith(prefix) for prefix in prefixes)]
    healthy = [path for path in matches if load_dataset_info(path).get("total_episodes", 0) > 0]
    healthy.sort(key=lambda path: path.stat().st_mtime)
    if max_datasets is None:
        return healthy
    return healthy[-max_datasets:]


def summarize_dataset(path: Path) -> tuple[int, int]:
    info = load_dataset_info(path)
    return int(info.get("total_episodes", 0)), int(info.get("total_frames", 0))


def latest_ac_from_files(files: list[str], base_file: str = DEFAULT_BASE_AC_FILE) -> ACRef:
    candidates = [
        name
        for name in files
        if name.startswith("ac_checkpoint/") and name.endswith("/rl_checkpoint.pt")
    ]
    if candidates:
        selected = sorted(candidates)[-1]
        version = selected.removeprefix("ac_checkpoint/").removesuffix("/rl_checkpoint.pt")
        return ACRef(version=version, hf_path=selected)
    version = Path(base_file).stem
    return ACRef(version=version, hf_path=base_file)


def resolve_latest_ac(model_repo: str, base_file: str = DEFAULT_BASE_AC_FILE) -> ACRef:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id=model_repo, repo_type="model")
    return latest_ac_from_files(files, base_file=base_file)


def emit_result(result: object) -> None:
    print(RESULT_PREFIX + json.dumps(asdict(result), sort_keys=True), flush=True)


def parse_result(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        raise ValueError(f"remote command did not emit {RESULT_PREFIX!r}; output tail:\n{stdout[-4000:]}")
    return json.loads(lines[-1].removeprefix(RESULT_PREFIX))


def run_checked(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[cmd] {printable}", flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True)


def run_capture(cmd: list[str]) -> str:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[remote] {printable}", flush=True)
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    print(proc.stdout, end="")
    return proc.stdout


def merge_datasets(source_dirs: list[Path], output_root: Path, merged_name: str, task: str) -> Path:
    from lerobot.datasets.aggregate import aggregate_datasets

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    aggregate_datasets(
        repo_ids=[f"local/{path.name}" for path in source_dirs],
        aggr_repo_id=f"local/{merged_name}",
        roots=source_dirs,
        aggr_root=output_root,
    )
    set_task_name(output_root, task)
    return output_root


def set_task_name(dataset_root: Path, task: str) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task"))
    pq.write_table(pa.Table.from_pandas(df), tasks_path)


def upload_private_dataset(dataset_root: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset_root),
        commit_message=f"Upload RLT online dataset {Path(repo_id).name}",
    )


def command_pack_upload(args: argparse.Namespace) -> PackResult:
    ts = args.timestamp or timestamp_now()
    ac = resolve_latest_ac(args.model_repo, base_file=args.base_ac_file)
    ac_version = args.ac_version or ac.version
    merged_name = args.dataset_name or f"{sanitize_name(ac_version)}_{ts}"
    repo_id = args.dataset_repo_id or f"{args.dataset_namespace}/{merged_name}"

    if args.source_dirs:
        source_dirs = [Path(path).expanduser().resolve() for path in args.source_dirs]
    else:
        source_dirs = discover_dataset_dirs(
            Path(args.datasets_root),
            prefixes=args.source_prefix,
            max_datasets=args.max_datasets,
        )
    if not source_dirs:
        raise ValueError("no source LeRobot datasets found for pack-upload")
    missing = [str(path) for path in source_dirs if not is_lerobot_dataset(path)]
    if missing:
        raise ValueError(f"not LeRobot dataset dirs: {missing}")

    output_root = Path(args.output_root).expanduser().resolve() / merged_name
    if args.dry_run:
        total_eps = sum(summarize_dataset(path)[0] for path in source_dirs)
        total_frames = sum(summarize_dataset(path)[1] for path in source_dirs)
        result = PackResult(repo_id, str(output_root), merged_name, ac_version, total_eps, total_frames)
        emit_result(result)
        return result

    print(f"[pack] merging {len(source_dirs)} datasets -> {output_root}", flush=True)
    for path in source_dirs:
        eps, frames = summarize_dataset(path)
        print(f"[pack]   {path} episodes={eps} frames={frames}", flush=True)
    merge_datasets(source_dirs, output_root, merged_name, args.task)
    upload_private_dataset(output_root, repo_id)
    total_eps, total_frames = summarize_dataset(output_root)
    result = PackResult(repo_id, str(output_root), merged_name, ac_version, total_eps, total_frames)
    emit_result(result)
    return result


def snapshot_model_base(model_repo: str, base_dir: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=model_repo,
        repo_type="model",
        allow_patterns=[f"{base_dir}/*", f"{base_dir}/**/*"],
        local_dir=str(cache_dir / "models" / sanitize_name(model_repo)),
    ))


def snapshot_dataset(dataset_repo_id: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        local_dir=str(cache_dir / "datasets" / sanitize_name(dataset_repo_id)),
    ))


def download_ac_checkpoint(model_repo: str, ac_ref: ACRef) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model_repo, repo_type="model", filename=ac_ref.hf_path))


def configure_from_ac_checkpoint(config, ac_ckpt: dict) -> None:
    from lerobot.rlt.utils import infer_actor_architecture
    from scripts.visualize_rlt_warmup_compare import infer_critic_architecture

    metadata = ac_ckpt.get("metadata", {}) or {}
    actor_meta = metadata.get("actor", {}) or {}
    critic_meta = metadata.get("critic", {}) or {}
    training_meta = metadata.get("training", {}) or {}
    actor_inferred = infer_actor_architecture(ac_ckpt["actor_state_dict"])
    critic_inferred = infer_critic_architecture(ac_ckpt["critic_state_dict"])

    config.chunk_length = int(metadata.get("chunk_length", config.chunk_length))
    config.action_dim = int(metadata.get("action_dim", config.action_dim))
    config.proprio_dim = int(metadata.get("proprio_dim", config.proprio_dim))
    config.actor.hidden_dim = int(actor_meta.get("hidden_dim", actor_inferred["hidden_dim"]))
    config.actor.num_layers = int(actor_meta.get("num_layers", actor_inferred["num_layers"]))
    config.actor.fixed_std = float(actor_meta.get("fixed_std", actor_inferred["fixed_std"]))
    config.actor.ref_dropout_p = float(actor_meta.get("ref_dropout_p", actor_inferred["ref_dropout_p"]))
    config.actor.activation = str(actor_meta.get("activation", actor_inferred["activation"]))
    config.actor.layer_norm = bool(actor_meta.get("layer_norm", actor_inferred["layer_norm"]))
    config.actor.residual = bool(actor_meta.get("residual", actor_inferred["residual"]))
    config.critic.hidden_dim = int(critic_meta.get("hidden_dim", critic_inferred["hidden_dim"]))
    config.critic.num_layers = int(critic_meta.get("num_layers", critic_inferred["num_layers"]))
    config.critic.activation = str(critic_meta.get("activation", critic_inferred["activation"]))
    config.critic.layer_norm = bool(critic_meta.get("layer_norm", critic_inferred["layer_norm"]))
    config.critic.residual = bool(critic_meta.get("residual", critic_inferred["residual"]))
    config.training.gamma = float(training_meta.get("gamma", config.training.gamma))
    config.training.beta = float(training_meta.get("beta", config.training.beta))
    config.training.tau = float(training_meta.get("tau", config.training.tau))
    config.training.utd_ratio = int(training_meta.get("utd_ratio", config.training.utd_ratio))
    config.training.actor_update_interval = int(
        training_meta.get("actor_update_interval", config.training.actor_update_interval)
    )


def build_transition_cache(
    args: argparse.Namespace,
    dataset_root: Path,
    model_snapshot: Path,
    out_dir: Path,
) -> None:
    script = REPO_ROOT / "scripts" / "rlt_training" / "build_transition_cache_v2.py"
    cmd = [
        sys.executable,
        str(script),
        "--demo-dataset-repo-id", args.dataset_repo_id,
        "--demo-dataset-root", str(dataset_root),
        "--rl-token-policy-path", str(model_snapshot / args.base_rlt_policy_dir),
        "--vla-pretrained-path", str(model_snapshot / args.base_dir),
        "--output-dir", str(out_dir),
        "--task-instruction", args.task,
        "--chunk-length", str(args.chunk_length),
        "--frame-stride", str(args.frame_stride),
        "--batch-size", str(args.encode_batch_size),
        "--num-workers", str(args.num_workers),
        "--train-ratio", str(args.train_ratio),
        "--device", args.device,
    ]
    if args.max_episodes is not None:
        cmd.extend(["--max-episodes", str(args.max_episodes)])
    run_checked(cmd, cwd=REPO_ROOT)


def train_from_cache(
    args: argparse.Namespace,
    ac_path: Path,
    cache_dir: Path,
    output_dir: Path,
    ac_ref: ACRef,
) -> Path:
    import torch
    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.trainer import offline_rl_loop
    from scripts.rlt_training.common import load_training_config
    from scripts.rlt_training.train_chunk_actor_critic import (
        create_algorithm_with_cached_transitions,
        load_cached_replay_buffers,
    )

    config = load_training_config(args.config)
    ac_ckpt = torch.load(ac_path, map_location="cpu", weights_only=False)
    configure_from_ac_checkpoint(config, ac_ckpt)
    config.offline_rl.num_gradient_steps = args.gradient_steps
    config.offline_rl.log_every = args.log_every
    config.offline_rl.eval_every = args.eval_every
    config.offline_rl.save_every = args.save_every
    config.training.batch_size = args.batch_size
    if args.beta is not None:
        config.training.beta = args.beta
    if args.actor_lr is not None:
        config.actor.lr = args.actor_lr
    if args.critic_lr is not None:
        config.critic.lr = args.critic_lr

    train_buffer, val_buffer = load_cached_replay_buffers(str(cache_dir), config.replay.capacity)
    algorithm = create_algorithm_with_cached_transitions(config, None, args.device)
    algorithm.policy.actor.load_state_dict(ac_ckpt["actor_state_dict"])
    algorithm.critic.load_state_dict(ac_ckpt["critic_state_dict"])
    algorithm.target_critic.load_state_dict(
        ac_ckpt.get("target_critic_state_dict", ac_ckpt["critic_state_dict"])
    )
    algorithm.target_actor.load_state_dict(ac_ckpt["actor_state_dict"])
    algorithm.to(args.device)

    actor_optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)
    metadata = {
        "base_ac_version": ac_ref.version,
        "base_ac_hf_path": ac_ref.hf_path,
        "dataset_repo_id": args.dataset_repo_id,
    }
    offline_rl_loop(
        algorithm=algorithm,
        config=config,
        replay_buffer=train_buffer,
        val_buffer=val_buffer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        save_dir=str(output_dir),
        metadata=metadata,
    )
    eval_metrics = evaluate_offline(algorithm, val_buffer, config, num_batches=args.eval_batches)
    (output_dir / "online_eval.json").write_text(json.dumps(asdict(eval_metrics), indent=2))
    return output_dir / "rl_checkpoint.pt"


def upload_ac_checkpoint(model_repo: str, checkpoint_dir: Path, ac_version: str) -> str:
    from huggingface_hub import HfApi

    remote_dir = f"ac_checkpoint/{ac_version}"
    api = HfApi()
    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        path_in_repo=remote_dir,
        commit_message=f"Upload RLT online AC checkpoint {ac_version}",
    )
    latest = {
        "version": ac_version,
        "checkpoint_hf_path": f"{remote_dir}/rl_checkpoint.pt",
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    latest_path = checkpoint_dir / "latest_ac_checkpoint.json"
    latest_path.write_text(json.dumps(latest, indent=2))
    api.upload_file(
        repo_id=model_repo,
        repo_type="model",
        path_or_fileobj=str(latest_path),
        path_in_repo="ac_checkpoint/latest.json",
        commit_message=f"Mark latest RLT online AC checkpoint {ac_version}",
    )
    return f"{remote_dir}/rl_checkpoint.pt"


def command_train_upload(args: argparse.Namespace) -> TrainResult:
    ts = args.timestamp or timestamp_now()
    output_root = Path(args.output_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    run_name = args.run_name or f"{sanitize_name(args.dataset_repo_id)}_{ts}"
    output_dir = output_root / run_name
    transition_cache = cache_root / "transition_cache" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    transition_cache.mkdir(parents=True, exist_ok=True)

    model_snapshot = snapshot_model_base(args.model_repo, args.base_dir, cache_root)
    dataset_root = snapshot_dataset(args.dataset_repo_id, cache_root)
    if args.base_ac_file_override:
        ac_ref = ACRef(args.base_ac_version, args.base_ac_file)
        ac_path = Path(args.base_ac_file_override)
    else:
        ac_ref = resolve_latest_ac(args.model_repo, args.base_ac_file)
        ac_path = download_ac_checkpoint(args.model_repo, ac_ref)

    print(f"[train] dataset={args.dataset_repo_id} root={dataset_root}", flush=True)
    print(f"[train] base_ac={ac_ref.version} path={ac_ref.hf_path}", flush=True)
    train_cache = transition_cache / "chunk_transitions_train.pt"
    val_cache = transition_cache / "chunk_transitions_val.pt"
    if train_cache.is_file() and val_cache.is_file():
        print(f"[train] reusing transition cache {transition_cache}", flush=True)
    else:
        build_transition_cache(args, dataset_root, model_snapshot, transition_cache)
    ckpt_path = train_from_cache(args, ac_path, transition_cache, output_dir, ac_ref)
    ac_version = args.ac_version or ts
    if args.skip_upload:
        checkpoint_hf_path = "SKIPPED"
    else:
        checkpoint_hf_path = upload_ac_checkpoint(args.model_repo, output_dir, ac_version)
    result = TrainResult(
        ac_version,
        checkpoint_hf_path,
        str(ckpt_path),
        str(output_dir),
        str(transition_cache),
    )
    emit_result(result)
    return result


def command_deploy_ac(args: argparse.Namespace) -> DeployResult:
    ac_ref = resolve_latest_ac(args.model_repo, args.base_ac_file)
    ckpt_path = download_ac_checkpoint(args.model_repo, ac_ref)
    deploy_dir = Path(args.deploy_dir).expanduser().resolve() / ac_ref.version
    deploy_dir.mkdir(parents=True, exist_ok=True)
    local_ckpt = deploy_dir / "rl_checkpoint.pt"
    shutil.copy2(ckpt_path, local_ckpt)
    latest = Path(args.latest_symlink).expanduser().resolve()
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = latest.with_suffix(".tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(local_ckpt)
    tmp_link.replace(latest)
    result = DeployResult(ac_ref.version, ac_ref.hf_path, str(local_ckpt), str(latest))
    emit_result(result)
    return result


def remote_command(repo: str, python_bin: str, role_args: list[str]) -> str:
    script = Path(repo) / "scripts" / "rlt_online_cycle.py"
    parts = ["cd", shlex.quote(repo), "&&", shlex.quote(python_bin), shlex.quote(str(script))]
    parts.extend(shlex.quote(part) for part in role_args)
    return " ".join(parts)

def command_cycle(args: argparse.Namespace) -> None:
    pack_args = [
        "pack-upload",
        "--datasets-root", args.robot_datasets_root,
        "--output-root", args.robot_merged_root,
        "--model-repo", args.model_repo,
        "--dataset-namespace", args.dataset_namespace,
        "--max-datasets", str(args.max_datasets),
        "--task", args.task,
    ]
    for prefix in args.source_prefix:
        pack_args.extend(["--source-prefix", prefix])
    pack_cmd = remote_command(args.robot_repo, args.robot_python, pack_args)
    pack_stdout = run_capture([args.ssh_connect, "c", "--cmd", pack_cmd])
    pack = parse_result(pack_stdout)

    train_args = [
        "train-upload",
        "--dataset-repo-id", pack["dataset_repo_id"],
        "--model-repo", args.model_repo,
        "--gradient-steps", str(args.gradient_steps),
        "--task", args.task,
    ]
    if args.train_max_episodes is not None:
        train_args.extend(["--max-episodes", str(args.train_max_episodes)])
    train_cmd = remote_command(args.coder_b_repo, args.coder_b_python, train_args)
    train_stdout = run_capture([args.coder_connect, "b", "--cmd", train_cmd])
    train = parse_result(train_stdout)

    deploy_args = [
        "deploy-ac",
        "--model-repo", args.model_repo,
        "--deploy-dir", args.robot_deploy_dir,
        "--latest-symlink", args.robot_latest_symlink,
    ]
    deploy_cmd = remote_command(args.robot_repo, args.robot_python, deploy_args)
    deploy_stdout = run_capture([args.ssh_connect, "c", "--cmd", deploy_cmd])
    deploy = parse_result(deploy_stdout)
    print(json.dumps({"pack": pack, "train": train, "deploy": deploy}, indent=2, sort_keys=True))


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--base-ac-file", default=DEFAULT_BASE_AC_FILE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack-upload", help="merge robot datasets and upload a private HF dataset")
    add_common_model_args(pack)
    pack.add_argument("--datasets-root", default=DEFAULT_ROBOT_DATASETS_ROOT)
    pack.add_argument("--source-dirs", nargs="*", default=[])
    pack.add_argument("--source-prefix", action="append", default=["eval_rlt_hil_wo_prefix_"])
    pack.add_argument("--max-datasets", type=int, default=None)
    pack.add_argument("--output-root", default="~/.cache/rlt_online/merged")
    pack.add_argument("--dataset-namespace", default="Shiki42")
    pack.add_argument("--dataset-name", default=None)
    pack.add_argument("--dataset-repo-id", default=None)
    pack.add_argument("--ac-version", default=None)
    pack.add_argument("--timestamp", default=None)
    pack.add_argument("--task", default=DEFAULT_TASK)
    pack.add_argument("--dry-run", action="store_true")
    pack.set_defaults(func=command_pack_upload)

    train = sub.add_parser("train-upload", help="train on coder b and upload AC ckpt")
    add_common_model_args(train)
    train.add_argument("--dataset-repo-id", required=True)
    train.add_argument("--base-rlt-policy-dir", default=DEFAULT_BASE_RLT_POLICY_DIR)
    train.add_argument("--base-ac-file-override", default=None)
    train.add_argument("--base-ac-version", default="local_override")
    train.add_argument("--output-root", default="/home/coder/share/outputs/rlt_online")
    train.add_argument("--cache-root", default="/home/coder/share/cache/rlt_online")
    train.add_argument("--run-name", default=None)
    train.add_argument("--ac-version", default=None)
    train.add_argument("--timestamp", default=None)
    train.add_argument("--config", default=None)
    train.add_argument("--device", default="cuda")
    train.add_argument("--task", default=DEFAULT_TASK)
    train.add_argument("--chunk-length", type=int, default=10)
    train.add_argument("--frame-stride", type=int, default=2)
    train.add_argument("--encode-batch-size", type=int, default=8)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--train-ratio", type=float, default=0.9)
    train.add_argument("--max-episodes", type=int, default=None)
    train.add_argument("--gradient-steps", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--save-every", type=int, default=100)
    train.add_argument("--eval-batches", type=int, default=10)
    train.add_argument("--beta", type=float, default=None)
    train.add_argument("--actor-lr", type=float, default=None)
    train.add_argument("--critic-lr", type=float, default=None)
    train.add_argument("--skip-upload", action="store_true")
    train.set_defaults(func=command_train_upload)

    deploy = sub.add_parser("deploy-ac", help="download newest AC ckpt on robot and update symlink")
    add_common_model_args(deploy)
    deploy.add_argument("--deploy-dir", default="~/rlt_online/ac_checkpoints")
    deploy.add_argument("--latest-symlink", default="~/rlt_online/latest_rl_checkpoint.pt")
    deploy.set_defaults(func=command_deploy_ac)

    cycle = sub.add_parser("cycle", help="local orchestrator: c pack -> b train -> c deploy")
    cycle.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    cycle.add_argument("--dataset-namespace", default="Shiki42")
    cycle.add_argument("--source-prefix", action="append", default=["eval_rlt_hil_wo_prefix_"])
    cycle.add_argument("--max-datasets", type=int, default=8)
    cycle.add_argument("--gradient-steps", type=int, default=200)
    cycle.add_argument("--train-max-episodes", type=int, default=None)
    cycle.add_argument("--task", default=DEFAULT_TASK)
    cycle.add_argument("--ssh-connect", default="/Users/shuyuan/.codex/skills/ssh/scripts/connect.sh")
    cycle.add_argument("--coder-connect", default="/Users/shuyuan/.codex/skills/coder/scripts/connect.sh")
    cycle.add_argument("--robot-repo", default=DEFAULT_ROBOT_REPO)
    cycle.add_argument("--coder-b-repo", default=DEFAULT_CODER_B_REPO)
    cycle.add_argument("--robot-python", default="python")
    cycle.add_argument("--coder-b-python", default="/home/coder/venv-lerobot/bin/python")
    cycle.add_argument("--robot-datasets-root", default=DEFAULT_ROBOT_DATASETS_ROOT)
    cycle.add_argument("--robot-merged-root", default="~/.cache/rlt_online/merged")
    cycle.add_argument("--robot-deploy-dir", default="~/rlt_online/ac_checkpoints")
    cycle.add_argument("--robot-latest-symlink", default="~/rlt_online/latest_rl_checkpoint.pt")
    cycle.set_defaults(func=command_cycle)
    return parser


def main() -> None:
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    os.environ.setdefault("HF_LEROBOT_HOME", str(Path.home() / ".cache" / "huggingface" / "lerobot"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_parser().parse_args()
    started = time.time()
    print(
        f"[rlt-online] command={args.command} "
        f"started={datetime.now().isoformat(timespec='seconds')}",
        flush=True,
    )
    args.func(args)
    print(f"[rlt-online] done elapsed_s={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
