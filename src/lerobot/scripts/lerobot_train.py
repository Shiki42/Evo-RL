#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import RLTokenJointConfig, TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler, build_weighted_dataset_sampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.acp_dataset_stats import compute_acp_indicator_stats
from lerobot.rl.acp_hook import build_acp_raw_batch_hook
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_rl_token_state,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


def _validate_rl_token_meta(
    meta: dict,
    rl_token: torch.nn.Module,
    cfg: TrainPipelineConfig,
) -> None:
    """Verify checkpoint meta.json matches the current RLTokenJointConfig + module shape."""
    module_cfg = meta.get("module_config", {})
    expected = {
        "token_dim": int(getattr(rl_token, "token_dim")),
        "num_rl_tokens": cfg.rl_token.num_rl_tokens,
        "num_enc_layers": cfg.rl_token.num_enc_layers,
        "num_dec_layers": cfg.rl_token.num_dec_layers,
        "ff_dim": cfg.rl_token.ff_dim,
    }
    mismatches = {
        key: (module_cfg.get(key), expected_value)
        for key, expected_value in expected.items()
        if module_cfg.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"rl_token checkpoint meta.json does not match cfg.rl_token: {mismatches}"
        )


def _maybe_build_rl_token(
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    device: torch.device,
) -> tuple[torch.nn.Module | None, int | None]:
    """Build RLTokenModule + compute num_image_tokens when cfg.rl_token.enable.

    Returns (None, None) when disabled. Validation has already enforced policy.type == 'pi05'
    and the absence of PEFT / compile_model.
    """
    if not cfg.rl_token.enable:
        return None, None

    from lerobot.rlt.rl_token import RLTokenModule

    unwrapped = policy
    if hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    pwx = unwrapped.model.paligemma_with_expert
    token_dim = pwx.paligemma.config.text_config.hidden_size
    rl_token = RLTokenModule(
        token_dim=token_dim,
        nhead=cfg.rl_token.nhead,
        num_enc_layers=cfg.rl_token.num_enc_layers,
        num_dec_layers=cfg.rl_token.num_dec_layers,
        ff_dim=cfg.rl_token.ff_dim,
        num_rl_tokens=cfg.rl_token.num_rl_tokens,
        inference_only=False,
    ).to(device)

    vision_cfg = pwx.paligemma.config.vision_config
    tokens_per_camera = (unwrapped.config.image_resolution[0] // vision_cfg.patch_size) ** 2
    num_image_tokens = tokens_per_camera * len(unwrapped.config.image_features)
    return rl_token, num_image_tokens


def _joint_forward(
    *,
    policy: PreTrainedPolicy,
    batch: Any,
    rabc_batch_weights,
    rabc_batch_stats,
    rl_token: torch.nn.Module,
    rl_token_cfg: RLTokenJointConfig,
    num_image_tokens: int | None,
) -> tuple[torch.Tensor, dict]:
    """Joint VLA + RL Token forward path. See docs/rlt/joint_train_plan.md section 4."""
    from lerobot.rlt.utils import postprocess_prefix_tokens

    if rl_token_cfg is None:
        raise ValueError("_joint_forward requires rl_token_cfg.")
    if num_image_tokens is None:
        raise ValueError("_joint_forward requires num_image_tokens.")

    if rabc_batch_weights is not None:
        per_sample_loss_vla, output_dict, prefix_hidden = policy.forward_with_prefix(
            batch, reduction="none"
        )
        epsilon = 1e-6
        loss_vla = (per_sample_loss_vla * rabc_batch_weights).sum() / (
            rabc_batch_weights.sum() + epsilon
        )
        output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
        output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
        output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
    else:
        loss_vla, output_dict, prefix_hidden = policy.forward_with_prefix(batch)

    prefix_for_rl = postprocess_prefix_tokens(
        prefix_hidden.to(dtype=torch.float32),
        image_only=rl_token_cfg.image_only,
        num_image_tokens=num_image_tokens,
        pool_size=rl_token_cfg.token_pool_size,
    )
    loss_recon = rl_token.reconstruction_loss(prefix_for_rl)
    loss = loss_vla + rl_token_cfg.weight * loss_recon

    output_dict["loss_vla"] = loss_vla.detach().float().item()
    output_dict["loss_recon"] = loss_recon.detach().float().item()
    output_dict["loss_total"] = loss.detach().float().item()
    output_dict["rl_token_weight"] = float(rl_token_cfg.weight)
    output_dict["rl_token_prefix_tokens"] = int(prefix_for_rl.shape[1])
    return loss, output_dict


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
    rl_token: torch.nn.Module | None = None,
    rl_token_cfg: RLTokenJointConfig | None = None,
    num_image_tokens: int | None = None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if rl_token is not None:
            loss, output_dict = _joint_forward(
                policy=policy,
                batch=batch,
                rabc_batch_weights=rabc_batch_weights,
                rabc_batch_stats=rabc_batch_stats,
                rl_token=rl_token,
                rl_token_cfg=rl_token_cfg,
                num_image_tokens=num_image_tokens,
            )
        elif rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if rl_token is not None:
        params_to_clip = list(policy.parameters()) + list(rl_token.parameters())
    else:
        params_to_clip = list(policy.parameters())

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(params_to_clip, grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params_to_clip, float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(
    cfg: TrainPipelineConfig,
    accelerator: Accelerator | None = None,
):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()
    acp_raw_batch_hook = build_acp_raw_batch_hook(cfg.acp, cfg.seed)

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        if cfg.acp.enable:
            indicator_stats = compute_acp_indicator_stats(dataset, cfg.acp.indicator_field)
            if indicator_stats is None:
                logging.warning(
                    "ACP is enabled but indicator statistics are unavailable for field '%s'.",
                    cfg.acp.indicator_field,
                )
            else:
                if indicator_stats.total_count >= 0:
                    logging.info(
                        "ACP indicator stats (%s): field='%s' ratio=%.6f positive=%d total=%d",
                        indicator_stats.source,
                        indicator_stats.indicator_field,
                        indicator_stats.positive_ratio,
                        indicator_stats.positive_count,
                        indicator_stats.total_count,
                    )
                else:
                    logging.info(
                        "ACP indicator stats (%s): field='%s' ratio=%.6f",
                        indicator_stats.source,
                        indicator_stats.indicator_field,
                        indicator_stats.positive_ratio,
                    )
                if indicator_stats.invalid_count > 0:
                    logging.warning(
                        "ACP indicator field '%s' contains %d non-binary values (expected only 0/1).",
                        indicator_stats.indicator_field,
                        indicator_stats.invalid_count,
                    )

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    rl_token, num_image_tokens = _maybe_build_rl_token(cfg, policy, device)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    if rl_token is not None:
        rl_token_lr = cfg.optimizer.lr * cfg.rl_token.lr_multiplier
        optimizer.add_param_group(
            {
                "params": [p for p in rl_token.parameters() if p.requires_grad],
                "lr": rl_token_lr,
                "name": "rl_token",
            }
        )
        # Pytorch LR schedulers built from a single param group don't auto-extend their internal
        # state when add_param_group is called later. Patch the cosine/warmup state in-place so
        # the rl_token group follows the same schedule as the policy group.
        if lr_scheduler is not None:
            if hasattr(lr_scheduler, "base_lrs"):
                lr_scheduler.base_lrs.append(rl_token_lr)
            if hasattr(lr_scheduler, "lr_lambdas") and lr_scheduler.lr_lambdas:
                lr_scheduler.lr_lambdas.append(lr_scheduler.lr_lambdas[0])
            if hasattr(lr_scheduler, "_last_lr"):
                lr_scheduler._last_lr.append(rl_token_lr)  # noqa: SLF001
        if is_main_process:
            logging.info(
                f"Joint RL Token training enabled: weight={cfg.rl_token.weight}, "
                f"num_rl_tokens={cfg.rl_token.num_rl_tokens}, "
                f"token_pool_size={cfg.rl_token.token_pool_size}, "
                f"image_only={cfg.rl_token.image_only}, lr={rl_token_lr}"
            )

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        if rl_token is not None:
            # Load rl_token state BEFORE accelerator.prepare (param identity preservation).
            meta = load_rl_token_state(cfg.checkpoint_path, rl_token, strict=True)
            _validate_rl_token_meta(meta, rl_token, cfg)
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames") and cfg.dataset.sampling_weights is not None:
        raise ValueError(
            "dataset.sampling_weights and EpisodeAwareSampler (triggered by "
            "policy.drop_n_last_frames) are mutually exclusive: pick one."
        )
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    elif cfg.dataset.sampling_weights is not None:
        shuffle = False
        if cfg.dataset.sampling_group_frames is None:
            raise ValueError("dataset.sampling_weights requires dataset.sampling_group_frames")
        if sum(cfg.dataset.sampling_group_frames) != len(dataset):
            raise ValueError(
                f"sampling_group_frames sum ({sum(cfg.dataset.sampling_group_frames)}) "
                f"must equal dataset length ({len(dataset)})"
            )
        sampler = build_weighted_dataset_sampler(
            cfg.dataset.sampling_group_frames, cfg.dataset.sampling_weights
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator. accelerator.prepare(None) is a no-op so passing rl_token=None
    # is safe; we still avoid the call when rl_token is None to keep enable=False byte-identical.
    accelerator.wait_for_everyone()
    if rl_token is not None:
        policy, optimizer, dataloader, lr_scheduler, rl_token = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, rl_token
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    logged_first_prompt = False
    prompt_keys: list[str] = []
    policy_task_field = getattr(cfg.policy, "task_field", None)
    if isinstance(policy_task_field, str) and policy_task_field:
        prompt_keys.append(policy_task_field)
    for key in ("task", "subtask"):
        if key not in prompt_keys:
            prompt_keys.append(key)

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        if acp_raw_batch_hook is not None:
            batch = acp_raw_batch_hook(batch, step)
        batch = preprocessor(batch)

        if is_main_process and not logged_first_prompt:
            for key in prompt_keys:
                if key not in batch:
                    continue
                prompt_batch = batch[key]
                first_prompt = None
                if isinstance(prompt_batch, str):
                    first_prompt = prompt_batch
                elif isinstance(prompt_batch, (list, tuple)) and len(prompt_batch) > 0:
                    first_item = prompt_batch[0]
                    if isinstance(first_item, str):
                        first_prompt = first_item
                if first_prompt is not None:
                    logging.info("First policy prompt (%s):\n%s", key, first_prompt)
                    logged_first_prompt = True
                    break
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
            rl_token=rl_token,
            rl_token_cfg=cfg.rl_token if rl_token is not None else None,
            num_image_tokens=num_image_tokens,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    rl_token=(accelerator.unwrap_model(rl_token) if rl_token is not None else None),
                    rl_token_cfg=(cfg.rl_token if rl_token is not None else None),
                    num_image_tokens=num_image_tokens,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
