"""R3 regression: _validate_vla_path pre-flight guards against baked path drift.

Without this guard, PI05Policy.from_pretrained(strict=False) would silently
random-init the backbone when vla_pretrained_path is missing or empty — a
catastrophic failure mode that produces a fully-random VLA in deploy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.policies.rlt.modeling_rlt_token import _validate_vla_path


def test_raises_when_path_does_not_exist(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="vla_pretrained_path does not exist"):
        _validate_vla_path(str(missing))


def test_raises_when_path_is_a_file_not_dir(tmp_path: Path):
    f = tmp_path / "looks_like_dir"
    f.write_text("oops")
    with pytest.raises(FileNotFoundError, match="vla_pretrained_path does not exist"):
        _validate_vla_path(str(f))


def test_raises_when_dir_has_no_weights(tmp_path: Path):
    # config.json present but no model weights — the case where a partial
    # release would slip through to PI05Policy.from_pretrained and silently
    # random-init.
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="No model weights"):
        _validate_vla_path(str(tmp_path))


def test_accepts_safetensors(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    out = _validate_vla_path(str(tmp_path))
    assert out == tmp_path


def test_accepts_sharded_safetensors(tmp_path: Path):
    (tmp_path / "model-00001-of-00003.safetensors").write_bytes(b"\x00")
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    out = _validate_vla_path(str(tmp_path))
    assert out == tmp_path


def test_accepts_pytorch_bin(tmp_path: Path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"\x00")
    out = _validate_vla_path(str(tmp_path))
    assert out == tmp_path
