import torch
from typing import Tuple, Optional
from . import _core

def dda_compute_fragments(
    resolution: Tuple[int, int],
    pos: torch.Tensor,         # (B, V, 4)
    edges: torch.Tensor,       # (E, 2)
    edge_ids: Optional[torch.Tensor] = None  # (N,), optional
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform software rasterization of a line mesh using DDA into fragments.

    Args:
        resolution (tuple[int, int]): Image resolution (H, W).
        pos (torch.Tensor): Homogeneous vertex positions (B, V, 4), float32 CUDA.
        edges (torch.Tensor): Edge indices (E, 2), int32 CUDA.
        edge_ids (torch.Tensor, optional): Edge indices to rasterize. Defaults to all edges.

    Returns:
        frag_pix (torch.Tensor): (N, 3) int32 tensor with (batch, h, w) per valid fragment.
        frag_attrs (torch.Tensor): (N, 4) float32 tensor with (axis, t1, z_ndc, edge_id+1).
    """
    H, W = resolution

    if not pos.is_cuda or not edges.is_cuda:
        raise ValueError("Input tensors must be on CUDA.")
    if pos.dtype != torch.float32 or edges.dtype != torch.int32:
        raise ValueError("`pos` must be float32 and `edges` must be int32.")
    if pos.ndim != 3 or pos.shape[2] != 4:
        raise ValueError("`pos` must have shape (B, V, 4).")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("`edges` must have shape (E, 2).")
    if not pos.is_contiguous() or not edges.is_contiguous():
        raise ValueError("Input tensors must be contiguous.")

    B, V, _ = pos.shape
    E = edges.shape[0]
    device = pos.device

    if edge_ids is None:
        edge_ids = torch.arange(B * E, dtype=torch.int32, device=device)
    else:
        if not edge_ids.is_cuda or edge_ids.dtype != torch.int32 or edge_ids.ndim != 1:
            raise ValueError("`edge_ids` must be a 1D int32 CUDA tensor.")

    num_prims = edge_ids.shape[0]

    # Allocate outputs for span computation
    frag_prefix = torch.empty((num_prims,), dtype=torch.int32, device=device)
    frag_slopes = torch.empty((num_prims, 2), dtype=torch.float32, device=device)
    frag_spans = torch.empty((num_prims, 4), dtype=torch.float32, device=device)

    num_frags = _core.dda_compute_span(
        edge_ids,
        H, W, pos, edges,
        frag_prefix, frag_slopes, frag_spans
    )

    frag_pix = torch.full((num_frags, 3), -1, dtype=torch.int32, device=device)
    frag_attrs = torch.empty((num_frags, 4), dtype=torch.float32, device=device)

    _core.dda_compute_fragments(
        frag_prefix, edge_ids,
        frag_slopes, frag_spans,
        H, W, pos, edges,
        frag_pix, frag_attrs
    )

    frag_pix_out = torch.empty_like(frag_pix)
    frag_attrs_out = torch.empty_like(frag_attrs)
    valid_count = _core.filter_valid_fragments(frag_pix, frag_attrs, frag_pix_out, frag_attrs_out)

    frag_pix = frag_pix_out[:valid_count]
    frag_attrs = frag_attrs_out[:valid_count]
    return frag_pix, frag_attrs


class DDAInterpolateAttributes(torch.autograd.Function):
    """
    Autograd wrapper for DDA-based attribute interpolation.

    Interpolates per-vertex attributes along rasterized lines using DDA, 
    and supports gradient backpropagation via atomic adds.
    """

    @staticmethod
    def forward(ctx, rast: torch.Tensor, attr: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if not rast.is_cuda or not attr.is_cuda or not edges.is_cuda:
            raise ValueError("All input tensors must be CUDA tensors.")
        if rast.dtype != torch.float32 or attr.dtype != torch.float32:
            raise ValueError("`rast` and `attr` must be float32.")
        if edges.dtype != torch.int32:
            raise ValueError("`edges` must be int32.")
        if rast.ndim != 4 or rast.shape[-1] != 4:
            raise ValueError("`rast` must have shape (B, H, W, 4).")
        if attr.ndim != 2:
            raise ValueError("`attr` must have shape (V, C).")
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("`edges` must have shape (E, 2).")

        B, H, W, _ = rast.shape
        V, C = attr.shape

        image = torch.zeros((B, H, W, C), dtype=torch.float32, device='cuda')
        _core.dda_interpolate_attributes(rast.contiguous(), attr.contiguous(), edges.contiguous(), image)

        ctx.save_for_backward(rast, attr, edges)
        return image

    @staticmethod
    def backward(ctx, d_image: torch.Tensor):
        rast, attr, edges = ctx.saved_tensors

        d_attr = torch.zeros_like(attr)
        _core.backward_dda_interpolate_attributes(
            rast.contiguous(), d_image.contiguous(), edges.contiguous(), d_attr
        )

        return None, d_attr, None  # Only attr has gradient


# Convenience wrapper
def dda_interpolate_attributes(rast: torch.Tensor, attr: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Interpolate per-vertex attributes for line fragments using DDA rasterization (with gradients).

    Args:
        rast (torch.Tensor): (B, H, W, 4) DDA raster buffer.
        attr (torch.Tensor): (V, C) per-vertex attributes.
        edges (torch.Tensor): (E, 2) edge indices.

    Returns:
        torch.Tensor: (B, H, W, C) interpolated image.
    """
    return DDAInterpolateAttributes.apply(rast, attr, edges)
