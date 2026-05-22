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
from collections.abc import Iterator

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def build_weighted_dataset_sampler(
    per_dataset_num_frames: list[int],
    weights: list[float],
) -> torch.utils.data.WeightedRandomSampler:
    """WeightedRandomSampler for a concatenated MultiLeRobotDataset with a fixed mix ratio.

    Each frame of sub-dataset d gets weight (weights[d]/sum(weights))/num_frames[d], so every
    sub-dataset contributes its target proportion regardless of size. Draws with replacement.

    Args:
        per_dataset_num_frames: Frame count of each sub-dataset, in concatenation order.
        weights: Desired sampling proportion per sub-dataset (need not sum to 1).

    Raises:
        ValueError: On length mismatch, an empty sub-dataset, or non-positive total weight.
    """
    if len(per_dataset_num_frames) != len(weights):
        raise ValueError(
            f"per_dataset_num_frames ({len(per_dataset_num_frames)}) and weights "
            f"({len(weights)}) length mismatch"
        )
    if any(n <= 0 for n in per_dataset_num_frames):
        raise ValueError(f"every sub-dataset must be non-empty, got {per_dataset_num_frames}")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"weights must be non-negative with positive sum, got {weights}")
    norm = sum(weights)
    total = sum(per_dataset_num_frames)
    sample_weights = torch.zeros(total, dtype=torch.double)
    start = 0
    for n, w in zip(per_dataset_num_frames, weights, strict=True):
        sample_weights[start : start + n] = (w / norm) / n
        start += n
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=total, replacement=True
    )
