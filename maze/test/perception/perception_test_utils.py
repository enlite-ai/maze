"""Utils used in perception block unit testing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def build_input_tensor(dims: Sequence[int]) -> torch.Tensor:
    """build input tensor"""
    np_array = np.random.randn(*dims).astype(np.float32)
    tensor = torch.from_numpy(np_array)
    return tensor


def build_input_dict(dims: Sequence[int]) -> dict[str, torch.Tensor]:
    """build input dictionary"""
    return {'in_key': build_input_tensor(dims)}


def build_multi_input_dict(dims: Sequence[Sequence[int]]) -> dict[str, torch.Tensor]:
    """build multi input dictionary"""
    input_dict = {}
    for i, d in enumerate(dims):
        input_dict[f'in_key_{i}'] = build_input_tensor(d)
    return input_dict
