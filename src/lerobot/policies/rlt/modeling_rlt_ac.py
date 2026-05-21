from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.rlt.action_modifier import PrefixOutputCapture, RLTActionModifier
from lerobot.policies.rlt.configuration_rlt_ac import ChunkACPolicyConfig
from lerobot.policies.rlt.modeling_rlt_token import RLTokenPolicy
from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.critic import TwinCritic
from lerobot.rlt.losses import actor_loss, critic_loss
from lerobot.rlt.phase_controller import PhaseController
from lerobot.rlt.utils import soft_update

log = logging.getLogger(__name__)


class ChunkACPolicy(PreTrainedPolicy):
    """Chunk-level TD3+BC offline RL policy with VLA reference.

    Holds two frozen backbones in __dict__ (so they don't land in safetensors):
      * `_rl_token_policy`: an RLTokenPolicy loaded from
        config.rl_token_pretrained_path. It internally holds the frozen pi0.5
        backbone, so we only need one stash for the whole VLA chain.

    Trainable modules (registered as nn submodules → in safetensors + optimizer):
      * `actor`: ChunkActor — refines or replaces the VLA reference chunk.
      * `critic`: TwinCritic — twin Q networks.
      * `target_critic`: TwinCritic — Polyak-averaged copy of critic.

    forward(batch) returns one scalar loss. Each forward == one critic update.
    Actor is updated every `actor_update_interval` calls. Target critic is
    soft-updated with `tau` after every critic step. UTD=k is achieved by
    setting lerobot-train `--steps` to outer*k.
    """

    config_class = ChunkACPolicyConfig
    name = "rlt_ac"

    def __init__(
        self,
        config: ChunkACPolicyConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: ChunkACPolicyConfig = config

        rl_token_policy = self._load_rl_token_policy()
        object.__setattr__(self, "_rl_token_policy", rl_token_policy)
        self._validate_rl_token_arch(rl_token_policy)

        state_dim = config.rl_token_dim + config.proprio_dim
        chunk_dim = config.chunk_length * config.action_dim

        self.actor = ChunkActor(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            fixed_std=config.actor_fixed_std,
            ref_dropout_p=config.actor_ref_dropout_p,
            activation=config.actor_activation,
            layer_norm=config.actor_layer_norm,
            residual=config.actor_residual,
        )
        self.critic = TwinCritic(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            activation=config.critic_activation,
            layer_norm=config.critic_layer_norm,
            residual=config.critic_residual,
        )
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.target_critic.eval()

        # Persistent step counter — survives ckpt save/load.
        self.register_buffer("_critic_step", torch.zeros((), dtype=torch.long), persistent=True)

        # Deploy-only: lazy build at .reset() time.
        self.modifier: RLTActionModifier | None = None
        # Deploy toggle (set by lerobot_rlt_record). False => the actor sees a
        # zeroed VLA reference chunk in RL phase (mirrors training ref-dropout).
        self.vla_ref: bool = True

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_rl_token_policy(self) -> RLTokenPolicy:
        if not self.config.rl_token_pretrained_path:
            raise ValueError(
                "ChunkACPolicy requires config.rl_token_pretrained_path "
                "pointing at a saved RLTokenPolicy checkpoint dir."
            )
        policy = RLTokenPolicy.from_pretrained(self.config.rl_token_pretrained_path)
        for p in policy.parameters():
            p.requires_grad = False
        policy.eval()
        return policy

    def _validate_rl_token_arch(self, rl_token_policy: RLTokenPolicy) -> None:
        rtp_cfg = rl_token_policy.config
        if rtp_cfg.rl_token_dim != self.config.rl_token_dim:
            raise ValueError(
                f"rl_token_dim mismatch: ChunkACPolicyConfig={self.config.rl_token_dim} vs "
                f"RLTokenPolicy ckpt={rtp_cfg.rl_token_dim}"
            )
        if rtp_cfg.rl_token_num_rl_tokens != self.config.rl_token_num_rl_tokens:
            raise ValueError(
                f"num_rl_tokens mismatch: ChunkACPolicyConfig={self.config.rl_token_num_rl_tokens} vs "
                f"RLTokenPolicy ckpt={rtp_cfg.rl_token_num_rl_tokens}"
            )

    # ------------------------------------------------------------------
    # Training: forward
    # ------------------------------------------------------------------

    def _coerce_batch(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Convert ChunkTransition-style batch into the dict expected by losses.

        Required keys: state_vec, exec_chunk, ref_chunk, reward_seq,
        next_state_vec, next_ref_chunk, done, actual_steps.
        Adds flattened views: exec_chunk_flat, ref_chunk_flat, next_ref_flat.
        """
        out: dict[str, Tensor] = {}
        for k in (
            "state_vec",
            "exec_chunk",
            "ref_chunk",
            "reward_seq",
            "next_state_vec",
            "next_ref_chunk",
            "done",
            "actual_steps",
        ):
            if k not in batch:
                raise KeyError(f"ChunkACPolicy.forward missing batch key: {k!r}")
            v = batch[k]
            if not isinstance(v, Tensor):
                v = torch.as_tensor(v)
            out[k] = v
        out["exec_chunk_flat"] = out["exec_chunk"].flatten(start_dim=-2)
        out["ref_chunk_flat"] = out["ref_chunk"].flatten(start_dim=-2)
        out["next_ref_flat"] = out["next_ref_chunk"].flatten(start_dim=-2)
        return out

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        tx = self._coerce_batch(batch)

        c_loss = critic_loss(
            self.critic,
            self.target_critic,
            self.actor,
            tx,
            gamma=self.config.gamma,
            C=self.config.chunk_length,
        )
        soft_update(self.target_critic, self.critic, self.config.tau)
        self._critic_step += 1

        do_actor = (int(self._critic_step.item()) % self.config.actor_update_interval) == 0
        if do_actor:
            a_loss = actor_loss(self.actor, self.critic, tx, beta=self.config.beta)
            total = c_loss + a_loss
            info = {
                "loss": total.detach(),
                "loss_critic": c_loss.detach(),
                "loss_actor": a_loss.detach(),
                "critic_step": self._critic_step.detach().clone(),
            }
            return total, info

        info = {
            "loss": c_loss.detach(),
            "loss_critic": c_loss.detach(),
            "critic_step": self._critic_step.detach().clone(),
        }
        return c_loss, info

    # ------------------------------------------------------------------
    # Inference: predict_action_chunk + select_action
    # ------------------------------------------------------------------

    def _ensure_modifier(self) -> RLTActionModifier:
        if self.modifier is None:
            phase_ctrl = self._build_phase_controller()
            rl_token_module = self._rl_token_policy.rl_token
            self.modifier = RLTActionModifier(
                rl_token=rl_token_module,
                actor=self.actor,
                phase_ctrl=phase_ctrl,
                chunk_length=self.config.chunk_length,
                action_dim=self.config.action_dim,
                proprio_dim=self.config.proprio_dim,
                chunk_exec_steps=self.config.chunk_exec_steps,
                vla_ref=self.vla_ref,
            )
            self._prefix_capture = PrefixOutputCapture(
                token_pool_size=self.config.token_pool_size,
                image_only=self.config.image_only,
                num_image_tokens=self._compute_num_image_tokens(),
            )
            self._prefix_capture.attach(self._rl_token_policy._pi05)
        return self.modifier

    def _build_phase_controller(self) -> PhaseController:
        # Bridge: ChunkACPolicyConfig.phase_mode encodes the deploy policy
        # (always_rl / always_vla / manual). PhaseController itself only
        # accepts manual or learned (= how transitions happen). For
        # always_* we still use manual mode and just pin the initial phase.
        mode = self.config.phase_mode
        if mode == "always_rl":
            ctrl = PhaseController(mode="manual")
            ctrl.trigger_critical()
            return ctrl
        if mode == "always_vla":
            ctrl = PhaseController(mode="manual")
            ctrl.trigger_vla()
            return ctrl
        if mode == "manual":
            return PhaseController(mode="manual")
        raise ValueError(f"Unknown phase_mode: {mode!r}")

    def _compute_num_image_tokens(self) -> int:
        pi05 = self._rl_token_policy._pi05
        pwx = pi05.model.paligemma_with_expert
        vision_cfg = pwx.paligemma.config.vision_config
        tokens_per_camera = (self.config.image_resolution[0] // vision_cfg.patch_size) ** 2
        return tokens_per_camera * len(self.config.camera_keys)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        mod = self._ensure_modifier()
        pi05 = self._rl_token_policy._pi05
        vla_chunk = pi05.predict_action_chunk(batch, **kwargs)
        vla_chunk = vla_chunk[:, :, : self.config.action_dim]
        prefix_tokens = self._prefix_capture.consume()
        proprio = batch["observation.state"][:, : self.config.proprio_dim]
        return mod.compute_chunk(vla_chunk, proprio, prefix_tokens)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        mod = self._ensure_modifier()
        if mod.needs_new_chunk:
            chunk = self.predict_action_chunk(batch, **kwargs)
            mod.enqueue(chunk)
        return mod.pop_action()

    def reset(self) -> None:
        if self.modifier is not None:
            self.modifier.reset()

    def get_optim_params(self) -> list:
        return [
            {"params": list(self.actor.parameters())},
            {"params": list(self.critic.parameters())},
        ]

    # ------------------------------------------------------------------
    # Device + train-mode plumbing for the stashed backbone
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._rl_token_policy.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self._rl_token_policy.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        self._rl_token_policy.cpu()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen backbones always in eval.
        self._rl_token_policy.eval()
        self.target_critic.eval()
        return self
