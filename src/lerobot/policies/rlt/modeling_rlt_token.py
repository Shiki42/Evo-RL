from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.rlt.configuration_rlt_token import RLTokenPolicyConfig
from lerobot.rlt.rl_token import RLTokenModule
from lerobot.rlt.utils import postprocess_prefix_tokens

log = logging.getLogger(__name__)


def _load_pi05_config_from_dir(pretrained_path: str) -> PI05Config:
    """Load a PI05Config from a ckpt dir, stripping the ``type`` polymorphic field.

    The fork's PI05Config does not declare ``type``; draccus chokes if it's
    present (which the SFT config.json always is). Strip and parse.
    """
    import draccus

    config_path = Path(pretrained_path) / "config.json"
    with open(config_path) as fh:
        raw = json.load(fh)
    raw.pop("type", None)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(raw, tmp)
        tmp_path = tmp.name
    try:
        with draccus.config_type("json"):
            return draccus.parse(PI05Config, tmp_path, args=[])
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _load_norm_stats(path: str | None) -> Tensor | None:
    """Load per-dim std for weighted reconstruction loss.

    Expected file format: torch.save({"std": Tensor[token_dim]}, path).
    Returns None when path is None.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"norm_stats_path does not exist: {path}")
    blob = torch.load(p, map_location="cpu")
    if isinstance(blob, dict) and "std" in blob:
        return blob["std"].to(dtype=torch.float32)
    if isinstance(blob, Tensor):
        return blob.to(dtype=torch.float32)
    raise ValueError(f"unrecognized norm_stats payload at {path}: keys={list(blob) if isinstance(blob, dict) else type(blob)}")


class RLTokenPolicy(PreTrainedPolicy):
    """Training-only policy that fits an RLTokenModule on top of frozen pi0.5.

    forward(batch) returns the reconstruction loss (+ optional pi0.5 supervised
    loss when vla_ft_weight > 0). The frozen pi0.5 backbone is loaded in __init__
    via PI05Policy.from_pretrained and stashed in self.__dict__ to bypass
    nn.Module's submodule registration — so it is NOT serialized into the
    saved safetensors and does NOT appear in get_optim_params() either.

    deploy is handled by ChunkACPolicy; this policy raises on inference paths.
    """

    config_class = RLTokenPolicyConfig
    name = "rlt_token"

    def __init__(
        self,
        config: RLTokenPolicyConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: RLTokenPolicyConfig = config

        self.rl_token = RLTokenModule(
            token_dim=config.rl_token_dim,
            nhead=config.rl_token_nhead,
            num_enc_layers=config.rl_token_enc_layers,
            num_dec_layers=config.rl_token_dec_layers,
            ff_dim=config.rl_token_ff_dim,
            num_rl_tokens=config.rl_token_num_rl_tokens,
            inference_only=False,
        )

        pi05 = self._load_pi05_backbone()
        # Stash pi0.5 OUTSIDE nn.Module submodule tracking. nn.Module.__setattr__
        # registers nn.Module values into self._modules; object.__setattr__ stores
        # in self.__dict__ instead — so state_dict() / get_optim_params() skip it.
        object.__setattr__(self, "_pi05", pi05)

        std = _load_norm_stats(config.norm_stats_path)
        if std is not None:
            self.register_buffer("_dim_std", std, persistent=False)
        else:
            self._dim_std = None  # type: ignore[assignment]

        self._num_image_tokens: int = self._compute_num_image_tokens(pi05)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_pi05_backbone(self) -> PI05Policy:
        pi05_cfg = _load_pi05_config_from_dir(self.config.vla_pretrained_path)
        pi05_cfg.dtype = self.config.vla_dtype
        pi05_cfg.device = self.config.device
        pi05 = PI05Policy.from_pretrained(
            self.config.vla_pretrained_path,
            config=pi05_cfg,
            revision=self.config.vla_revision,
            strict=False,
        )
        if self.config.vla_ft_weight == 0:
            for p in pi05.parameters():
                p.requires_grad = False
            pi05.eval()
        return pi05

    def _compute_num_image_tokens(self, pi05: PI05Policy) -> int:
        pwx = pi05.model.paligemma_with_expert
        vision_cfg = pwx.paligemma.config.vision_config
        tokens_per_camera = (self.config.image_resolution[0] // vision_cfg.patch_size) ** 2
        num_cameras = len(self.config.camera_keys)
        return tokens_per_camera * num_cameras

    # ------------------------------------------------------------------
    # PreTrainedPolicy abstract methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        pass

    def get_optim_params(self) -> list:
        groups = [
            {"params": list(self.rl_token.parameters())},
        ]
        if self.config.vla_ft_weight > 0:
            groups.append({"params": [p for p in self._pi05.parameters() if p.requires_grad]})
        return groups

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Reconstruction loss + optional pi0.5 supervised loss."""
        if self.config.vla_ft_weight > 0:
            loss_vla, info, prefix_hidden = self._pi05.forward_with_prefix(batch, reduction="mean")
        else:
            with torch.no_grad():
                _, info, prefix_hidden = self._pi05.forward_with_prefix(batch, reduction="mean")
            loss_vla = torch.zeros((), device=prefix_hidden.device, dtype=torch.float32)

        prefix_for_rl = postprocess_prefix_tokens(
            prefix_hidden.to(dtype=torch.float32),
            image_only=self.config.image_only,
            num_image_tokens=self._num_image_tokens,
            pool_size=self.config.token_pool_size,
            num_per_camera=self.config.num_per_camera,
            active_camera_indices=self.config.active_camera_indices,
        )

        loss_recon = self.rl_token.reconstruction_loss(
            prefix_for_rl,
            dim_std=self._dim_std,
            gamma=self.config.norm_gamma,
        )
        total = self.config.recon_weight * loss_recon + self.config.vla_ft_weight * loss_vla
        return total, {
            "loss": total.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_vla": loss_vla.detach(),
        }

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        raise NotImplementedError(
            "RLTokenPolicy is training-only; use ChunkACPolicy for inference."
        )

    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        raise NotImplementedError(
            "RLTokenPolicy is training-only; use ChunkACPolicy for inference."
        )

    # ------------------------------------------------------------------
    # Device + train-mode plumbing
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._pi05.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self._pi05.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        self._pi05.cpu()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.vla_ft_weight > 0:
            self._pi05.train(mode)
        else:
            self._pi05.eval()
        return self
