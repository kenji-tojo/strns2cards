import torch
from typing import Tuple, Any
from . import _core
from .dda import dda_compute_fragments
from .bitset_utils import popcount_bitset


def mark_discontinuity_edges(
    resolution: Tuple[int, int],
    pos: torch.Tensor,
    tri: torch.Tensor,
    edges: torch.Tensor,
    edge2tri: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute valid edges and normals for discontinuity detection (e.g., silhouette or boundary edges).

    Args:
        resolution (tuple[int, int]): (H, W) image resolution.
        pos (torch.Tensor): (B, V, 4) float32 vertex positions (homogeneous coordinates).
        tri (torch.Tensor): (T, 3) int32 triangle indices.
        edges (torch.Tensor): (E, 2) int32 edge list.
        edge2tri (torch.Tensor): (E, 2) int32 triangle adjacency per edge.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (prim_ids, normals)
            - prim_ids: (N,) int32 valid edge indices.
            - normals: (B*E, 2) float32 edge normals (all edges).
    """
    H, W = resolution
    device = pos.device

    B = pos.shape[0]
    E = edges.shape[0]

    prim_ids = torch.full((B * E,), -1, dtype=torch.int32, device=device)
    normals = torch.zeros((B * E, 2), dtype=torch.float32, device=device)

    _core.mark_discontinuity_edges(
        H, W,
        tri.contiguous(),
        edges.contiguous(),
        edge2tri.contiguous(),
        pos.contiguous(),
        prim_ids,
        normals
    )

    prim_ids_out = torch.empty_like(prim_ids)
    valid_count = _core.compact_valid_ints(prim_ids.contiguous(), prim_ids_out)
    prim_ids = prim_ids_out[:valid_count]
    return prim_ids, normals


class _AntialiasedClusterBitsetLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, loss_value: torch.Tensor, pos: torch.Tensor, d_pos: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(d_pos)
        return loss_value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:
        d_pos, = ctx.saved_tensors
        return None, grad_output * d_pos, None

def antialiased_cluster_bitset_loss(
    bitset: torch.Tensor,
    target_bitset: torch.Tensor,
    pos: torch.Tensor,
    tri: torch.Tensor,
    edges: torch.Tensor,
    edge2tri: torch.Tensor,
    edge2cluster: torch.Tensor,
    kernel_radius: float = 7.0,
    rho: float = 1.0,
    count_xor_error: bool = False,
) -> torch.Tensor:
    """
    Compute the antialiased cluster bitset loss and its gradient with respect to vertex positions.

    Args:
        bitset (torch.Tensor): (B, H, W, num_slots) int32 predicted bitset.
        target_bitset (torch.Tensor): (B, H, W, num_slots) int32 target bitset.
        pos (torch.Tensor): (B, V, 4) float32 homogeneous vertex positions.
        tri (torch.Tensor): (T, 3) int32 triangle indices.
        edges (torch.Tensor): (E, 2) int32 edge list.
        edge2tri (torch.Tensor): (E, 2) int32 triangle adjacency per edge.
        edge2cluster (torch.Tensor): (E,) int32 mapping edge index to cluster ID.
        kernel_radius (float, optional): Gaussian smoothing radius. Default: 7.0.
        rho (float, optional): Penalty scaling for false positives. Default: 1.0.
        count_xor_error (bool, optional): Whether to compute the loss by XOR popcount between bitsets. Default: False.

    Returns:
        torch.Tensor: Scalar loss (float32).
    """
    device = bitset.device
    assert target_bitset.device == device
    assert pos.device == device
    assert tri.device == device
    assert edges.device == device
    assert edge2tri.device == device
    assert edge2cluster.device == device

    _, H, W, _ = target_bitset.shape

    prim_ids, normals = mark_discontinuity_edges((H, W), pos, tri, edges, edge2tri)
    frag_pix, frag_attrs_dda = dda_compute_fragments((H, W), pos, edges, prim_ids)

    d_pos = torch.zeros_like(pos)
    _core.backward_antialiased_cluster_bitset(
        frag_pix.contiguous(),
        frag_attrs_dda.contiguous(),
        bitset.contiguous(),
        target_bitset.contiguous(),
        pos.contiguous(),
        edges.contiguous(),
        normals.contiguous(),
        edge2cluster.contiguous(),
        d_pos,
        kernel_radius,
        rho
    )

    if count_xor_error:
        bitset_xor = torch.bitwise_xor(bitset, target_bitset)
        loss_value = popcount_bitset(bitset_xor).sum()
    else:
        loss_value = torch.ones([], dtype=pos.dtype, device=device)

    return _AntialiasedClusterBitsetLossFn.apply(loss_value, pos, d_pos)
