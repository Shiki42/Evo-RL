"""Unit tests for build_weighted_dataset_sampler (multi-dataset 1:1 mixing)."""
from __future__ import annotations

from lerobot.datasets.sampler import build_weighted_dataset_sampler


def _count_per_dataset(sampler, per_dataset_num_frames):
    bounds, acc = [], 0
    for n in per_dataset_num_frames:
        bounds.append((acc, acc + n))
        acc += n
    counts = [0] * len(per_dataset_num_frames)
    for idx in sampler:
        for d, (lo, hi) in enumerate(bounds):
            if lo <= idx < hi:
                counts[d] += 1
                break
    return counts


def test_balances_1to1_despite_size_imbalance():
    num_frames = [8000, 2000]  # 4:1 size imbalance
    sampler = build_weighted_dataset_sampler(num_frames, [0.5, 0.5])
    counts = _count_per_dataset(sampler, num_frames)
    frac0 = counts[0] / sum(counts)
    assert 0.46 < frac0 < 0.54, f"expected ~0.5, got {frac0}"


def test_respects_uneven_ratio():
    num_frames = [1000, 1000]
    sampler = build_weighted_dataset_sampler(num_frames, [0.75, 0.25])
    counts = _count_per_dataset(sampler, num_frames)
    frac0 = counts[0] / sum(counts)
    assert 0.70 < frac0 < 0.80, f"expected ~0.75, got {frac0}"


def test_rejects_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        build_weighted_dataset_sampler([100, 200], [1.0])


def test_rejects_empty_dataset():
    import pytest
    with pytest.raises(ValueError):
        build_weighted_dataset_sampler([100, 0], [0.5, 0.5])


if __name__ == "__main__":
    test_balances_1to1_despite_size_imbalance()
    test_respects_uneven_ratio()
    for args in ([[100, 200], [1.0]], [[100, 0], [0.5, 0.5]]):
        try:
            build_weighted_dataset_sampler(*args)
            raise AssertionError(f"expected ValueError for {args}")
        except ValueError:
            pass
    print("all weighted-sampler tests passed")
