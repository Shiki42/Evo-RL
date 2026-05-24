from __future__ import annotations

from lerobot.policies.rlt.configuration_rlt_token import RLTokenPolicyConfig


def test_rlt_token_defaults_match_c12_architecture() -> None:
    cfg = RLTokenPolicyConfig()

    assert cfg.rl_token_num_rl_tokens == 1
    assert cfg.token_pool_size == 0
    assert cfg.image_only is False
    assert cfg.rl_token_nhead == 16
    assert cfg.rl_token_enc_layers == 4
    assert cfg.rl_token_dec_layers == 4
    assert cfg.rl_token_ff_dim == 8192
    assert cfg.norm_gamma == 0.25


def test_rlt_token_scheduler_matches_c12_run_length() -> None:
    scheduler = RLTokenPolicyConfig().get_scheduler_preset()

    assert scheduler is not None
    assert scheduler.peak_lr == 2e-4
    assert scheduler.decay_lr == 5e-6
    assert scheduler.num_warmup_steps == 200
    assert scheduler.num_decay_steps == 60000
