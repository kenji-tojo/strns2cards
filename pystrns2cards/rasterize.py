import torch
import math
from typing import Optional, Tuple
from . import _core

def compute_fragments(
    resolution: Tuple[int, int],
    pos: torch.Tensor,  # (B, V, 4)
    tri: torch.Tensor   # (T, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform software rasterization of a triangle mesh into fragments with CUDA timing.

    Args:
        resolution (tuple[int, int]): Image resolution (H, W).
        pos (torch.Tensor): Homogeneous vertex positions (B, V, 4), float32 CUDA.
        tri (torch.Tensor): Triangle indices (T, 3), int32 CUDA.

    Returns:
        frag_pix (torch.Tensor): (N, 3) int32 tensor with (batch, h, w) per valid fragment.
        frag_attrs (torch.Tensor): (N, 4) float32 tensor with (bary0, bary1, z, triangle_id+1).
    """
    H, W = resolution
    if not pos.is_cuda or not tri.is_cuda:
        raise ValueError("Input tensors must be on CUDA.")
    if pos.dtype != torch.float32 or tri.dtype != torch.int32:
        raise ValueError("`pos` must be float32 and `tri` must be int32.")
    if pos.ndim != 3 or pos.shape[2] != 4:
        raise ValueError("`pos` must have shape (B, V, 4).")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("`tri` must have shape (T, 3).")
    if not pos.is_contiguous() or not tri.is_contiguous():
        raise ValueError("Input tensors must be contiguous.")

    B, V, _ = pos.shape
    T = tri.shape[0]
    device = pos.device

    rects = torch.empty((B * T, 4), dtype=torch.int32, device=device)
    frag_prefix = torch.empty((B * T), dtype=torch.int32, device=device)
    num_frags = _core.compute_triangle_rects(H, W, pos, tri, rects, frag_prefix)

    frag_pix = torch.full((num_frags, 3), -1, dtype=torch.int32, device=device)
    frag_attrs = torch.empty((num_frags, 4), dtype=torch.float32, device=device)

    _core.compute_fragments(H, W, pos, tri, frag_prefix, rects, frag_pix, frag_attrs)

    frag_pix_out = torch.empty_like(frag_pix)
    frag_attrs_out = torch.empty_like(frag_attrs)
    valid_count = _core.filter_valid_fragments(frag_pix, frag_attrs, frag_pix_out, frag_attrs_out)
    frag_pix = frag_pix_out[:valid_count]
    frag_attrs = frag_attrs_out[:valid_count]
    return frag_pix, frag_attrs

def depth_test(
    B: int,
    resolution: Tuple[int, int],
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor
) -> torch.Tensor:
    """
    Depth test using z-buffering to keep the closest fragment per pixel.

    Args:
        B (int): Number of batches.
        resolution (tuple[int, int]): Image resolution (H, W).
        frag_pix (torch.Tensor): Fragment pixel indices of shape (N, 3), int32, CUDA.
                                 Format: [batch_idx, h, w]
        frag_attrs (torch.Tensor): Per-fragment attributes of shape (N, 4), float32, CUDA.
                                   Format: [bary0, bary1, zw, triangle_id + 1]

    Returns:
        torch.Tensor: Rasterized output of shape (B, H, W, 4), float32, CUDA.

    Raises:
        ValueError: If input tensors are invalid.
    """
    if not (frag_pix.is_cuda and frag_attrs.is_cuda):
        raise ValueError("Input tensors must be on CUDA.")
    if frag_pix.dtype != torch.int32:
        raise ValueError("frag_pix must be int32.")
    if frag_attrs.dtype != torch.float32:
        raise ValueError("frag_attrs must be float32.")
    if frag_pix.ndim != 2 or frag_pix.shape[1] != 3:
        raise ValueError("frag_pix must have shape (N, 3).")
    if frag_attrs.ndim != 2 or frag_attrs.shape[1] != 4:
        raise ValueError("frag_attrs must have shape (N, 4).")
    if frag_pix.shape[0] != frag_attrs.shape[0]:
        raise ValueError("frag_pix and frag_attrs must match in length.")

    H, W = resolution
    rast_out = torch.zeros((B, H, W, 4), dtype=torch.float32, device='cuda')

    _core.depth_test(H, W, frag_pix.contiguous(), frag_attrs.contiguous(), rast_out)
    return rast_out

def rasterize(
    resolution: Tuple[int, int],
    pos: torch.Tensor,
    tri: torch.Tensor
) -> torch.Tensor:
    """
    Perform full software rasterization with depth testing.

    Args:
        resolution (tuple[int, int]): Image resolution (H, W).
        pos (torch.Tensor): Homogeneous vertex positions (B, V, 4), float32 CUDA.
        tri (torch.Tensor): Triangle indices (T, 3), int32 CUDA.

    Returns:
        torch.Tensor: Rasterized output (B, H, W, 4), float32 CUDA.
                      Each pixel stores (bary0, bary1, z, triangle_id+1).
    """
    frag_pix, frag_attrs = compute_fragments(resolution, pos, tri)
    B = pos.shape[0]
    return depth_test(B, resolution, frag_pix, frag_attrs)


class InterpolateTriangleAttributes(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rast: torch.Tensor,
        attr: torch.Tensor,
        tri: torch.Tensor
    ) -> torch.Tensor:
        if not (rast.is_cuda and attr.is_cuda and tri.is_cuda):
            raise ValueError("All inputs must be CUDA tensors.")

        B, H, W, _ = rast.shape
        V, C = attr.shape
        image = torch.zeros((B, H, W, C), dtype=attr.dtype, device=attr.device)

        _core.interpolate_triangle_attributes(rast.contiguous(), attr.contiguous(), tri.contiguous(), image)

        ctx.save_for_backward(rast, tri, attr)
        return image

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        rast, tri, attr = ctx.saved_tensors

        d_attr = torch.zeros_like(attr)
        _core.backward_interpolate_triangle_attributes(
            rast.contiguous(), grad_output.contiguous(), tri.contiguous(), d_attr
        )

        return None, d_attr, None


def interpolate_triangle_attributes(
    rast: torch.Tensor,
    attr: torch.Tensor,
    tri: torch.Tensor
) -> torch.Tensor:
    """
    Differentiable wrapper for barycentric attribute interpolation.

    Args:
        rast: (B, H, W, 4) raster output
        attr: (V, C) vertex attributes
        tri: (T, 3) triangle indices

    Returns:
        (B, H, W, C) interpolated per-pixel output
    """
    return InterpolateTriangleAttributes.apply(rast, attr, tri)


def cluster_mask_from_fragments(
    resolution: Tuple[int, int],
    frag_pix: torch.Tensor,         # (N, 3), int32
    frag_attrs: torch.Tensor,       # (N, 4), float32
    tri2cluster: torch.Tensor,      # (T,), int32
    cluster_count: Optional[int] = None
) -> torch.Tensor:
    """
    Aggregate triangle fragments into a per-pixel cluster bitmask.

    Args:
        resolution (tuple[int, int]): Image resolution (H, W).
        frag_pix (torch.Tensor): (N, 3) int32 tensor with (batch, h, w).
        frag_attrs (torch.Tensor): (N, 4) float32 tensor with (bary0, bary1, z, triangle_id+1).
        tri2cluster (torch.Tensor): (T,) int32 mapping from triangle to cluster ID.
        cluster_count (int, optional): Total number of cluster IDs. If None, it is deduced from tri2cluster.max() + 1.

    Returns:
        torch.Tensor: (B, H, W, num_slots) int32 tensor. Each pixel stores a bitmask of visible clusters.
    """
    H, W = resolution

    if frag_pix.dtype != torch.int32 or frag_pix.ndim != 2 or frag_pix.shape[1] != 3:
        raise ValueError("frag_pix must be a (N, 3) int32 tensor.")
    if frag_attrs.dtype != torch.float32 or frag_attrs.ndim != 2 or frag_attrs.shape[1] != 4:
        raise ValueError("frag_attrs must be a (N, 4) float32 tensor.")
    if tri2cluster.dtype != torch.int32 or tri2cluster.ndim != 1:
        raise ValueError("tri2cluster must be a (T,) int32 tensor.")
    if not (frag_pix.is_cuda and frag_attrs.is_cuda and tri2cluster.is_cuda):
        raise ValueError("All input tensors must be CUDA tensors.")

    B = int(frag_pix[:, 0].max().item()) + 1
    cluster_count = cluster_count if cluster_count is not None else int(tri2cluster.max().item()) + 1
    num_slots = (cluster_count + 31) // 32

    bitset = torch.zeros((B, H, W, num_slots), dtype=torch.int32, device=frag_pix.device)

    _core.cluster_mask_from_fragments(
        frag_pix.contiguous(),
        frag_attrs.contiguous(),
        tri2cluster.contiguous(),
        bitset
    )

    return bitset
