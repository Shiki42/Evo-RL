"""Unit tests for scripts/rlt_training/build_transition_cache_v2.py.

Regression tests for three bugs found by review (2026-05-21):
  - B1: exec_chunk was the VLA prediction (==ref_chunk), should be teleop ground truth
  - B2: next_state was at idx+1 (== frame_stride away), should be at frame_to_idx[t+C]
  - B3: reward_seq was all-zero, should be sparse +1 via build_reward_seq

These tests mock pi0.5, the RL Token, the preprocessor, and the capture so the
script's dict-assembly logic can be exercised on CPU without GPUs or HF assets.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch
from torch.utils.data import Dataset


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts/rlt_training/build_transition_cache_v2.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("btc_v2", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["btc_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

_TELEOP_VALUE = 0.1
_VLA_VALUE = 0.7
_PROPRIO_DIM = 3
_ACTION_DIM = 2
_CHUNK = 4
_TOKEN_DIM = 5


class _SynthDataset(Dataset):
    """Each frame returns (observation.state filled with float(idx), teleop
    action chunk filled with _TELEOP_VALUE + idx*1e-3, task string).

    The per-row idx sentinel lets the tests check next_state alignment.
    """

    def __init__(self, n_frames: int):
        self.n = n_frames

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {
            "observation.state": torch.full((_PROPRIO_DIM,), float(idx)),
            "action": torch.full((_CHUNK, _ACTION_DIM), _TELEOP_VALUE) + idx * 1e-3,
            "task": "test",
        }


class _IdentityPreprocessor:
    def __call__(self, batch):
        return batch


class _MockCapture:
    """consume() returns a placeholder prefix sized to the most recent batch.
    `_last_B` is set by _MockPi05.predict_action_chunk so consume() always
    matches the current batch (last batch in a loader may be smaller).
    """

    def __init__(self):
        self._last_B = 1

    def consume(self):
        return torch.zeros(self._last_B, 1, 8)


class _MockPi05:
    """predict_action_chunk returns a constant _VLA_VALUE chunk distinct from teleop.
    Also updates the shared capture's _last_B so consume() returns matching shape.
    """

    def __init__(self, capture: _MockCapture):
        self._capture = capture

    def predict_action_chunk(self, pre):
        B = pre["observation.state"].shape[0]
        self._capture._last_B = B
        return torch.full((B, _CHUNK, _ACTION_DIM), _VLA_VALUE)


class _MockRLToken:
    def encode(self, prefix):
        B = prefix.shape[0]
        return torch.zeros(B, _TOKEN_DIM)


def _build_kwargs(
    mod,
    frame_indices: list[int],
    episode_last_frame: int,
    *,
    ep_id: int = 0,
    episode_success: bool = True,
    stride: int = 2,
    batch_size: int = 2,
):
    n_frames = (max(frame_indices) + 1) if frame_indices else 1
    ds = _SynthDataset(n_frames)
    capture = _MockCapture()
    return {
        "pi05": _MockPi05(capture),
        "rl_token": _MockRLToken(),
        "preprocessor": _IdentityPreprocessor(),
        "capture": capture,
        "dataset": ds,
        "frame_indices": frame_indices,
        "episode_last_frame": episode_last_frame,
        "chunk_length": _CHUNK,
        "action_dim": _ACTION_DIM,
        "proprio_dim": _PROPRIO_DIM,
        "batch_size": batch_size,
        "num_workers": 0,
        "device": "cpu",
        "empty_cache_every": 999,
        "task_str": "test",
        "ep_id": ep_id,
        "episode_success": episode_success,
        "stride": stride,
    }


# ---------------------------------------------------------------------------
# B1 regression: exec_chunk must be teleop, not VLA
# ---------------------------------------------------------------------------

def test_exec_chunk_is_teleop_not_vla():
    mod = _load_script()
    # build_overlap_frame_indices output for ep_start=0, ep_stop=9, C=4, stride=2:
    # raw stride anchors {0,2,4,6,8} -> filter to start anchors {0,2,4} (need +C<=8)
    # plus their +C frames {4,6,8} -> sorted union = [0,2,4,6,8]
    frames = [0, 2, 4, 6, 8]
    txs = mod._encode_episode(**_build_kwargs(mod, frames, episode_last_frame=8))

    assert len(txs) >= 1, f"expected at least one transition, got {len(txs)}"
    for t in txs:
        # exec is teleop sentinel ~0.1; ref is VLA sentinel 0.7. Must differ.
        assert (t["exec_chunk"] != t["ref_chunk"]).any(), \
            "B1 regression: exec_chunk must NOT equal ref_chunk"
        # exec near 0.1 (allow per-row offset of up to idx*1e-3)
        assert t["exec_chunk"].abs().mean() < 0.2, \
            f"exec_chunk mean too large: {t['exec_chunk'].mean()}"
        # ref exactly 0.7 from the mock
        assert torch.allclose(t["ref_chunk"], torch.full_like(t["ref_chunk"], _VLA_VALUE))


# ---------------------------------------------------------------------------
# B2 regression: next_state at t+C, not t+stride
# ---------------------------------------------------------------------------

def test_next_state_uses_plus_C_not_plus_stride():
    mod = _load_script()
    frames = [0, 2, 4, 6, 8]  # stride=2, C=4
    txs = mod._encode_episode(**_build_kwargs(mod, frames, episode_last_frame=8))

    # _SynthDataset puts float(row_idx) in observation.state, so the proprio
    # tail of state_vec encodes which encoded row it came from.
    # transition[0] is anchored at frame 0; its next_state must be x_{0+C}=x_4.
    # frames.index(4) == 2, so the proprio tail should be all 4.0s.
    proprio_tail_first = txs[0]["next_state_vec"][-_PROPRIO_DIM:]
    expected_first = torch.full((_PROPRIO_DIM,), 4.0)
    assert torch.allclose(proprio_tail_first, expected_first), (
        f"B2 regression: first tx next_state proprio={proprio_tail_first.tolist()} "
        f"expected {expected_first.tolist()} (x_{{0+C=4}}, NOT x_{{0+stride=2}})"
    )

    # transition[1] is anchored at frame 2; next_state must be x_{2+C}=x_6.
    proprio_tail_second = txs[1]["next_state_vec"][-_PROPRIO_DIM:]
    expected_second = torch.full((_PROPRIO_DIM,), 6.0)
    assert torch.allclose(proprio_tail_second, expected_second), (
        f"B2 regression: second tx next_state proprio={proprio_tail_second.tolist()} "
        f"expected {expected_second.tolist()}"
    )


# ---------------------------------------------------------------------------
# B3 regression: reward_seq sparse terminal +1, zeros elsewhere
# ---------------------------------------------------------------------------

def test_reward_seq_is_sparse_terminal_only():
    mod = _load_script()
    frames = [0, 2, 4, 6, 8]
    txs = mod._encode_episode(**_build_kwargs(mod, frames, episode_last_frame=8))

    sums = [t["reward_seq"].sum().item() for t in txs]
    # Exactly the last transition should sum to 1.0; all others 0.
    assert sums[-1] == 1.0, f"B3 regression: terminal reward sum={sums[-1]}"
    assert all(s == 0.0 for s in sums[:-1]), \
        f"B3 regression: non-terminal chunks have reward: {sums}"
    # +1 lands exactly at the last step of the terminal chunk.
    assert txs[-1]["reward_seq"][-1].item() == 1.0
    assert txs[-1]["reward_seq"][:-1].sum().item() == 0.0


def test_failure_episode_gives_zero_reward_everywhere():
    mod = _load_script()
    frames = [0, 2, 4, 6, 8]
    txs = mod._encode_episode(**_build_kwargs(
        mod, frames, episode_last_frame=8, episode_success=False,
    ))
    for t in txs:
        assert t["reward_seq"].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Safety check: success episode with no terminal chunk raises
# ---------------------------------------------------------------------------

def test_raises_when_success_episode_has_no_terminal_chunk():
    mod = _load_script()
    # Hand-craft frame_indices where no start_frame + C reaches episode_last_frame.
    # episode_last_frame=10, C=4. Valid terminal anchor would be at frame 6 (6+4=10).
    # Construct frames missing the anchor at 6:
    # frames=[0,4,8] -> 0+4=4 OK (next_idx=1), 4+4=8 OK (next_idx=2), 8+4=12>10 skip.
    # Neither 4 nor 8 hits last_frame=10, so no terminal chunk created -> raise.
    frames = [0, 4, 8]
    with pytest.raises(ValueError, match="no terminal chunk"):
        mod._encode_episode(**_build_kwargs(
            mod, frames, episode_last_frame=10, episode_success=True,
        ))


# ---------------------------------------------------------------------------
# Other invariants worth pinning
# ---------------------------------------------------------------------------

def test_done_flag_set_only_on_terminal_chunk():
    mod = _load_script()
    frames = [0, 2, 4, 6, 8]
    txs = mod._encode_episode(**_build_kwargs(mod, frames, episode_last_frame=8))
    dones = [int(t["done"].item()) for t in txs]
    assert sum(dones) == 1, f"exactly one terminal expected, got dones={dones}"
    assert dones[-1] == 1


def test_actual_steps_equals_chunk_length():
    mod = _load_script()
    frames = [0, 2, 4, 6, 8]
    txs = mod._encode_episode(**_build_kwargs(mod, frames, episode_last_frame=8))
    for t in txs:
        assert int(t["actual_steps"].item()) == _CHUNK


def test_chunk_shapes_match_contract():
    mod = _load_script()
    frames = [0, 2, 4, 6, 8]
    txs = mod._encode_episode(**_build_kwargs(mod, frames, episode_last_frame=8))
    for t in txs:
        assert t["exec_chunk"].shape == (_CHUNK, _ACTION_DIM)
        assert t["ref_chunk"].shape == (_CHUNK, _ACTION_DIM)
        assert t["next_ref_chunk"].shape == (_CHUNK, _ACTION_DIM)
        assert t["reward_seq"].shape == (_CHUNK,)
        # state_vec = z_rl(token_dim) + proprio(proprio_dim)
        assert t["state_vec"].shape == (_TOKEN_DIM + _PROPRIO_DIM,)
        assert t["next_state_vec"].shape == (_TOKEN_DIM + _PROPRIO_DIM,)


def test_empty_frame_indices_returns_empty():
    mod = _load_script()
    txs = mod._encode_episode(**_build_kwargs(mod, [], episode_last_frame=0))
    assert txs == []


def test_raises_on_bad_stride():
    mod = _load_script()
    frames = [0, 2, 4, 6, 8]
    with pytest.raises(ValueError, match="stride must be positive"):
        mod._encode_episode(**_build_kwargs(
            mod, frames, episode_last_frame=8, stride=0,
        ))
    with pytest.raises(ValueError, match="must be divisible by stride"):
        mod._encode_episode(**_build_kwargs(
            mod, frames, episode_last_frame=8, stride=3,  # 4 % 3 != 0
        ))
