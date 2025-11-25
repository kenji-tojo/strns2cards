import torch
from typing import Tuple

from . import _core  # Assumes compiled nanobind module is named `_core`

def accumulate_bitset_rgb(
    bitset: torch.Tensor,              # (B, H, W, num_slots), int32, CUDA
    cluster_rgb: torch.Tensor         # (num_clusters, 3), float32, CUDA, values ∈ [0, 1]
) -> torch.Tensor:
    """
    Render a color image by accumulating RGB colors based on cluster bitset.

    Args:
        bitset (torch.Tensor): Cluster membership bitset (B, H, W, num_slots).
        cluster_rgb (torch.Tensor): RGB lookup for each cluster ID (num_clusters, 3).

    Returns:
        torch.Tensor: Accumulated RGB color image (B, H, W, 3), float32 CUDA.
    """
    if not bitset.is_cuda or not cluster_rgb.is_cuda:
        raise ValueError("All inputs must be on CUDA")
    if bitset.dtype != torch.int32:
        raise TypeError("bitset must be int32")
    if cluster_rgb.dtype != torch.float32:
        raise TypeError("cluster_rgb must be float32")
    if cluster_rgb.shape[1] != 3:
        raise ValueError("cluster_rgb must have shape (num_clusters, 3)")

    B, H, W, num_slots = bitset.shape
    accum_color = torch.zeros((B, H, W, 3), dtype=torch.float32, device=bitset.device)
    _core.accumulate_bitset_rgb(bitset.contiguous(), cluster_rgb.contiguous(), accum_color)
    return accum_color


def popcount_bitset(
    bitset: torch.Tensor              # (B, H, W, num_slots), int32, CUDA
) -> torch.Tensor:
    """
    Count the number of active clusters per pixel.

    Args:
        bitset (torch.Tensor): Cluster membership bitset (B, H, W, num_slots).

    Returns:
        torch.Tensor: Per-pixel count of active bits (B, H, W), float32 CUDA.
    """
    if not bitset.is_cuda:
        raise ValueError("bitset must be on CUDA")
    if bitset.dtype != torch.int32:
        raise TypeError("bitset must be int32")

    B, H, W, num_slots = bitset.shape
    count_image = torch.zeros((B, H, W), dtype=torch.float32, device=bitset.device)
    _core.popcount_bitset(bitset.contiguous(), count_image)
    return count_image
