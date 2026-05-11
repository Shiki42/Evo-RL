"""Build chunk-transition cache using the new RLTokenPolicy (lerobot-format).

Replaces scripts/rlt_training/build_transition_cache.py which used the legacy
RLT .pt format. New flow:
  1. Load RLTokenPolicy.from_pretrained(rl_token_pretrained_path)
  2. Reuse the policy's saved preprocessor (matches SFT pi05 byte-for-byte)
  3. Iterate LeRobotDataset episode-by-episode, build overlapping chunks
  4. For each chunk: run pi0.5.predict_action_chunk -> capture prefix tokens
     -> rl_token.encode -> state_vec
  5. Save list[dict] to chunk_transitions_{train,val}.pt
"""
from __future__ import annotations

import argparse
import pathlib
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rlt.action_modifier import PrefixOutputCapture
from lerobot.policies.rlt.modeling_rlt_token import RLTokenPolicy
from lerobot.rlt.offline_dataset import build_overlap_frame_indices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dataset-repo-id", required=True)
    p.add_argument("--demo-dataset-root", required=True)
    p.add_argument("--rl-token-policy-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-instruction", default="screw")
    p.add_argument("--chunk-length", type=int, default=10)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap on episodes to process (debug).")
    p.add_argument("--tokenizer-override", default="google/paligemma-3b-pt-224")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _encode_episode(
    policy: RLTokenPolicy,
    preprocessor,
    capture: PrefixOutputCapture,
    dataset: LeRobotDataset,
    frame_indices: list[int],
    chunk_length: int,
    action_dim: int,
    proprio_dim: int,
    batch_size: int,
    device: str,
    task_str: str,
) -> list[dict[str, Tensor]]:
    """Encode every base frame in `frame_indices` and assemble adjacent-frame
    transitions. Capture must already be attached to policy._pi05."""
    pi05 = policy._pi05
    rl_token = policy.rl_token
    out: list[dict[str, Tensor]] = []
    if not frame_indices:
        return out

    loader = DataLoader(
        Subset(dataset, frame_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    state_vecs: list[Tensor] = []
    ref_chunks: list[Tensor] = []
    for batch_i, batch in enumerate(loader):
        if "task" not in batch:
            batch["task"] = [task_str] * batch["observation.state"].shape[0]
        elif isinstance(batch["task"], list) and not batch["task"]:
            batch["task"] = [task_str] * batch["observation.state"].shape[0]
        pre = preprocessor(batch)
        with torch.no_grad():
            vla_chunk = pi05.predict_action_chunk(pre)
            prefix = capture.consume()
            z = rl_token.encode(prefix.to(torch.float32))
        if z.dim() == 3:
            z = z.mean(dim=1)
        proprio = pre["observation.state"][:, :proprio_dim].detach().to("cpu")
        state_vec = torch.cat([z.detach().to("cpu"), proprio], dim=-1)
        ref_chunk = vla_chunk[:, :chunk_length, :action_dim].detach().to("cpu")
        state_vecs.append(state_vec)
        ref_chunks.append(ref_chunk)
        # Explicit GPU cleanup to avoid fragmentation across many batches.
        del vla_chunk, prefix, z, pre
        if (batch_i + 1) % 4 == 0:
            torch.cuda.empty_cache()

    state_vecs_t = torch.cat(state_vecs, dim=0)
    ref_chunks_t = torch.cat(ref_chunks, dim=0)

    N = state_vecs_t.shape[0]
    C = chunk_length
    for i in range(N - 1):
        is_last = i == (N - 2)
        next_i = i + 1
        out.append(
            {
                "state_vec": state_vecs_t[i],
                "exec_chunk": ref_chunks_t[i],
                "ref_chunk": ref_chunks_t[i],
                "reward_seq": torch.zeros(C, dtype=torch.float32),
                "next_state_vec": state_vecs_t[next_i],
                "next_ref_chunk": ref_chunks_t[next_i],
                "done": torch.tensor(float(is_last)),
                "intervention": torch.tensor(0.0),
                "actual_steps": torch.tensor(C, dtype=torch.int64),
                "source": torch.tensor(0, dtype=torch.int64),
                "episode_id": torch.tensor(0, dtype=torch.int64),
                "is_critical": torch.tensor(1.0),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] RLTokenPolicy from {args.rl_token_policy_path}")
    policy = RLTokenPolicy.from_pretrained(args.rl_token_policy_path).to(args.device).eval()
    cfg = policy.config

    print("[load] preprocessor (reuses pi05 pipeline, tokenizer override applied)")
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=args.rl_token_policy_path,
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": args.tokenizer_override}},
    )

    print(f"[load] LeRobotDataset {args.demo_dataset_repo_id} from {args.demo_dataset_root}")
    delta = {"action": [i / 30.0 for i in range(cfg.chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=args.demo_dataset_repo_id,
        root=args.demo_dataset_root,
        delta_timestamps=delta,
    )
    n_episodes = dataset.num_episodes
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)
    print(f"[info] episodes: {n_episodes} (total {dataset.num_episodes})")

    capture = PrefixOutputCapture(
        token_pool_size=cfg.token_pool_size,
        image_only=cfg.image_only,
        num_image_tokens=policy._num_image_tokens,
    )
    capture.attach(policy._pi05)
    try:
        ep_indices = list(range(n_episodes))
        random.shuffle(ep_indices)
        n_train = int(args.train_ratio * n_episodes)
        train_eps = ep_indices[:n_train]
        val_eps = ep_indices[n_train:]
        print(f"[split] train={len(train_eps)} val={len(val_eps)}")

        for split_name, eps in (("train", train_eps), ("val", val_eps)):
            all_tx: list[dict[str, Tensor]] = []
            for k, ep_id in enumerate(eps):
                ep_meta = dataset.meta.episodes
                ep_from = int(ep_meta["dataset_from_index"][ep_id])
                ep_to = int(ep_meta["dataset_to_index"][ep_id])
                frame_indices = build_overlap_frame_indices(
                    episode_start=ep_from,
                    episode_stop=ep_to,
                    chunk_length=cfg.chunk_size,
                    stride=args.frame_stride,
                )
                if k % 20 == 0:
                    print(f"  [{split_name}] ep {k}/{len(eps)} (id={ep_id}, frames={ep_to-ep_from}, chunks={len(frame_indices)})")
                ep_tx = _encode_episode(
                    policy=policy,
                    preprocessor=preprocessor,
                    capture=capture,
                    dataset=dataset,
                    frame_indices=frame_indices,
                    chunk_length=args.chunk_length,
                    action_dim=cfg.action_dim,
                    proprio_dim=cfg.proprio_dim,
                    batch_size=args.batch_size,
                    device=args.device,
                    task_str=args.task_instruction,
                )
                all_tx.extend(ep_tx)
            save_path = out_dir / f"chunk_transitions_{split_name}.pt"
            print(f"[save] {split_name}: {len(all_tx)} transitions -> {save_path}")
            torch.save(all_tx, save_path)
    finally:
        capture.detach()

    print("[done]")


if __name__ == "__main__":
    main()
