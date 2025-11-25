import numpy as np
import torch
from typing import Tuple, List
from tqdm import tqdm
from numpy.typing import NDArray
from scipy.spatial import KDTree
from . import _core

def minimal_rotation(T_src: torch.Tensor, T_dst: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimal rotation matrix that aligns T_src to T_dst.
    Args:
        T_src (torch.Tensor): Source tangent vectors of shape (*, 3).
        T_dst (torch.Tensor): Destination tangent vectors, broadcastable to T_src.
    Returns:
        torch.Tensor: Rotation matrices of shape (*, 3, 3).
    """
    # Ensure the inputs are float tensors
    T_src = T_src.float()
    T_dst = T_dst.float()

    # Normalize the input vectors
    T_src = T_src / T_src.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    T_dst = T_dst / T_dst.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Compute cross product (axis of rotation)
    axis = torch.cross(T_src, T_dst, dim=-1)

    # Compute sine and cosine of the angle
    sin_theta = axis.norm(dim=-1, keepdim=True)
    cos_theta = (T_src * T_dst).sum(dim=-1, keepdim=True)
    cos_theta = cos_theta.clamp(min=-1.0, max=1.0)

    # Avoid division by zero
    axis = axis / sin_theta.clamp(min=1e-8)

    # Construct skew-symmetric matrix for axis
    x, y, z = axis.unbind(-1)
    K = torch.zeros((*axis.shape[:-1], 3, 3), dtype=axis.dtype, device=axis.device)
    K[..., 0, 1] = -z
    K[..., 0, 2] = y
    K[..., 1, 0] = z
    K[..., 1, 2] = -x
    K[..., 2, 0] = -y
    K[..., 2, 1] = x

    # Handle edge cases
    identity = torch.eye(3, device=axis.device).expand_as(K)
    # Parallel case: cos(theta) = 1
    parallel_mask = (cos_theta > 1 - 1e-6).squeeze(-1)
    R = identity.clone()

    # Anti-parallel case: cos(theta) = -1
    antiparallel_mask = (cos_theta < -1 + 1e-6).squeeze(-1)
    canonical_180 = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], device=axis.device).expand(R[antiparallel_mask].shape)
    R[antiparallel_mask] = canonical_180

    # Compute Rodrigues' rotation for general case
    valid_mask = ~(parallel_mask | antiparallel_mask)
    if valid_mask.any():
        sin_theta = sin_theta[valid_mask]
        cos_theta = cos_theta[valid_mask]
        axis = axis[valid_mask]
        K = K[valid_mask]
        I = identity[valid_mask]
        sin_theta = sin_theta[..., None]
        cos_theta = cos_theta[..., None]
        R[valid_mask] = I + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)

    return R

def tangent_frame(tangent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute tangent frame (orthonormal basis) given a set of z-axis directions.

    Args:
        tangent (torch.Tensor): tangent direction of the frame (shape = (*, 3)).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: normal and binormal directions of the frame (shape = (*, 3)).
    """
    # Normalize ez to ensure it is a unit vector
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Extract components
    z = tangent[..., 2]
    sign = torch.sign(z)
    sign[sign == 0.0] = 1.0

    # Coefficients for constructing ex and ey
    a = -1.0 / (sign + z)
    b = tangent[..., 0] * tangent[..., 1] * a

    # Construct local frame
    normal = torch.stack([1.0 + sign * tangent[..., 0] * tangent[..., 0] * a, sign * b, -sign * tangent[..., 0]], dim=-1)
    binormal = torch.stack([b, sign + tangent[..., 1] * tangent[..., 1] * a, -tangent[..., 1]], dim=-1)

    return normal, binormal

def compute_bishop_frame(strands: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Bishop frame (Rotation Minimizing Frame) using the double reflection method.

    Args:
        strands: (N, S, 3) tensor of 3D points representing the strands.

    Returns:
        Tuple containing:
        - T: (N, S, 3) tangent vectors
        - N_out: (N, S, 3) normal vectors
        - B_out: (N, S, 3) binormal vectors
    """
    N, S, _ = strands.shape

    # Compute tangents
    tangents = strands[:, 1:, :] - strands[:, :-1, :]  # (N, S-1, 3)
    T = torch.zeros_like(strands)
    T[:, :-1, :] += tangents
    T[:, 1:, :] += tangents
    T = T / T.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Compute initial normal and binormal using tangent_frame()
    T0 = T[:, 0, :]
    N0, B0 = tangent_frame(T0)

    # Initialize output buffers
    N_out = torch.zeros_like(strands)
    B_out = torch.zeros_like(strands)
    N_out[:, 0, :] = N0
    B_out[:, 0, :] = B0

    # Process all subsequent points to update normals and binormals
    for i in range(1, S):
        v1 = strands[:, i, :] - strands[:, i - 1, :]
        # c1 = (v1 ** 2).sum(dim=-1, keepdim=True)
        c1 = (v1 ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)

        n_prev = N_out[:, i - 1, :]
        t_prev = T[:, i - 1, :]
        t_curr = T[:, i, :]

        # Lateral adjustment for normals and tangents
        nL = n_prev - 2 * (torch.sum(v1 * n_prev, dim=-1, keepdim=True) / c1) * v1
        tL = t_prev - 2 * (torch.sum(v1 * t_prev, dim=-1, keepdim=True) / c1) * v1

        v2 = t_curr - tL
        # c2 = (v2 ** 2).sum(dim=-1, keepdim=True)
        c2 = (v2 ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Adjusted normal
        n1 = nL - 2 * (torch.sum(v2 * nL, dim=-1, keepdim=True) / c2) * v2
        n1 = n1 / n1.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Binormal computation: B = T × N
        b1 = torch.cross(t_curr, n1, dim=-1)
        b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        N_out[:, i, :] = n1
        B_out[:, i, :] = b1

    return T, N_out, B_out

def compute_arclength(strands: torch.Tensor) -> torch.Tensor:
    """
    Compute the cumulative arclength along each strand.

    Args:
        strands (torch.Tensor): Input strands of shape (N, V, 3).

    Returns:
        torch.Tensor: Arclengths of shape (N, V).
    """
    N, V, _ = strands.shape
    deltas = torch.norm(strands[:, 1:] - strands[:, :-1], dim=2)
    arclength = torch.cat([torch.zeros((N, 1), device=strands.device), torch.cumsum(deltas, dim=1)], dim=1)
    return arclength

def smooth_strands(
    strands: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to strands using a CUDA kernel.

    Args:
        strands (torch.Tensor): Input strand points of shape (N, V, 3), must be CUDA tensor.
        beta (float): Smoothing parameter for the Gaussian kernel.

    Returns:
        torch.Tensor: Smoothed strand points of the same shape as input.
    """
    if not strands.is_cuda:
        raise ValueError("smooth_strands: Input tensor must be on CUDA.")

    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("smooth_strands: Input tensor must have shape (N, V, 3).")

    arclength = compute_arclength(strands)
    smoothed_strands = torch.zeros_like(strands)

    _core.smooth_strands(strands, arclength, smoothed_strands, beta)
    return smoothed_strands


def skinning(
    strands: torch.Tensor,
    handle_strands: torch.Tensor,
    canonical_handles: torch.Tensor,
    R: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Apply skinning transformation to strands using per-point rotation matrices and Gaussian weights.

    Args:
        strands (torch.Tensor): Input strand points of shape (N, V, 3), must be CUDA tensor.
        handle_strands (torch.Tensor): Handle strands of shape (N, V, 3), must be CUDA tensor.
        canonical_handles (torch.Tensor): Canonical handle positions of shape (N, V, 3), must be CUDA tensor.
        R (torch.Tensor): Per-point rotation matrices of shape (N, V, 3, 3), must be CUDA tensor.
        beta (float): Smoothing parameter for the Gaussian kernel.

    Returns:
        torch.Tensor: Skinned strand points of the same shape as input.
    """
    if not (strands.is_cuda and handle_strands.is_cuda and canonical_handles.is_cuda and R.is_cuda):
        raise ValueError("skinning: All input tensors must be on CUDA.")

    N, V, _ = strands.shape
    if handle_strands.shape != (N, V, 3) or canonical_handles.shape != (N, V, 3) or R.shape != (N, V, 3, 3):
        raise ValueError("skinning: Input tensor shapes do not match the expected dimensions.")

    arclength = compute_arclength(strands)
    skinned_strands = torch.zeros_like(strands)

    _core.skinning(strands, arclength, handle_strands, canonical_handles, R, skinned_strands, beta)
    return skinned_strands

def compute_aligned_strands(
    strands: torch.Tensor,
    roots: torch.Tensor,
    T_dst: torch.Tensor,
) -> torch.Tensor:
    """
    Align strands to a target direction and anchor them at the given root positions.

    Args:
        strands (torch.Tensor): Input strand points of shape (N, V, 3).
        roots (torch.Tensor): Root positions of shape (N, 3).
        T_dst (torch.Tensor): Target tangent direction of shape (1, 3).

    Returns:
        torch.Tensor: Aligned strand points of the same shape as input.
    """
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape (N, V, 3)")
    if roots.ndim != 2 or roots.shape[-1] != 3:
        raise ValueError("roots must have shape (N, 3)")
    if T_dst.ndim != 2 or T_dst.shape != (1, 3):
        raise ValueError("T_dst must have shape (1, 3)")

    deltas = torch.norm(strands[:, 1:] - strands[:, :-1], dim=2)
    cumsum_y = torch.cat([torch.zeros((strands.shape[0], 1), device=strands.device), torch.cumsum(deltas, dim=1)], dim=1)

    aligned_strands = cumsum_y[:, :, None] * T_dst[None, :, :]
    aligned_strands += roots[:, None, :]
    return aligned_strands

def compute_alignment_rotation(
    strands: torch.Tensor,
    T_dst: torch.Tensor
) -> torch.Tensor:
    """
    Compute the alignment rotation matrices for strands to align their frames to the target tangent direction.

    Args:
        strands (torch.Tensor): Input strand points of shape (N, V, 3).
        T_dst (torch.Tensor): Target tangent direction of shape (1, 3).

    Returns:
        torch.Tensor: Alignment rotation matrices of shape (N, V, 3, 3).
    """
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape (N, V, 3)")
    if T_dst.ndim != 2 or T_dst.shape != (1, 3):
        raise ValueError("T_dst must have shape (1, 3)")

    T_src, N_src, B_src = compute_bishop_frame(strands)
    T_dst = T_dst.expand_as(T_src)

    T_src_root = T_src[:, 0, :]
    N_src_root = N_src[:, 0, :]
    B_src_root = B_src[:, 0, :]

    R_align = minimal_rotation(T_src_root, T_dst[:, 0, :])
    N_dst_root = (R_align @ N_src_root.unsqueeze(-1)).squeeze(-1)
    B_dst_root = (R_align @ B_src_root.unsqueeze(-1)).squeeze(-1)

    N_dst = N_dst_root.unsqueeze(1).expand_as(N_src)
    B_dst = B_dst_root.unsqueeze(1).expand_as(B_src)

    R_src = torch.stack([T_src, N_src, B_src], dim=-1)  # (N, V, 3, 3)
    R_dst = torch.stack([T_dst, N_dst, B_dst], dim=-1)   # (N, V, 3, 3)
    R = torch.matmul(R_dst, R_src.transpose(-1, -2))  # (N, V, 3, 3)
    return R

def strands_to_tube(strands: np.ndarray, radius=1.0, n_circle=8, attr=None):
    """
    Convert a batch of strands into open-ended triangle mesh tubes using RMF.

    Args:
        strands: (N, S, 3) array of strand positions.
        radius: float, radius of the tube.
        n_circle: int, number of points per circle cross-section.
        attr: Optional (N, S, num_channels) attribute to be expanded to per-vertex.

    Returns:
        verts: (N*S*n_circle, 3) vertex array.
        faces: (M, 3) triangle face array.
        expanded_attr: (N*S*n_circle, num_channels) attribute array (if attr is provided).
    """
    assert strands.ndim == 3 and strands.shape[2] == 3, f"Expected (N, S, 3), got {strands.shape}"
    N, S, _ = strands.shape

    # Compute RMF
    T, N_, B_ = compute_bishop_frame(torch.from_numpy(strands))
    T, N_, B_ = T.cpu().numpy(), N_.cpu().numpy(), B_.cpu().numpy()

    # Circle template
    theta = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n_circle, 2)

    # Offset directions: (N, n_circle, S, 3)
    dirs = (circle[:, 0, None] * N_[:, :, None, :] +
            circle[:, 1, None] * B_[:, :, None, :])
    dirs = np.transpose(dirs, (0, 2, 1, 3))  # (N, S, n_circle, 3)

    # Vertex positions: (N, S, n_circle, 3) → (N*S*n_circle, 3)
    verts = strands[:, None, :, :] + radius * dirs  # (N, 1, S, 3) + (N, S, n_circle, 3)
    verts = verts.transpose(0, 2, 1, 3).reshape(N * S * n_circle, 3)

    # Build face template for 1 strand
    idx = np.arange(S * n_circle).reshape(S, n_circle)
    a = idx[:-1, :]
    b = np.roll(idx[:-1, :], -1, axis=1)
    c = idx[1:, :]
    d = np.roll(idx[1:, :], -1, axis=1)

    faces = np.stack([
        np.stack([a, c, d], axis=-1),
        np.stack([a, d, b], axis=-1)
    ], axis=0).reshape(-1, 3)

    # Offset faces per strand
    offsets = (np.arange(N) * S * n_circle)[:, None, None]  # (N, 1, 1)
    faces_all = faces[None, :, :] + offsets
    faces_all = faces_all.reshape(-1, 3).astype(np.int32)

    # Attribute expansion (optional)
    expanded_attr = None
    if attr is not None:
        assert attr.shape[:2] == (N, S), f"Expected attr shape (N, S, C), got {attr.shape}"
        num_channels = attr.shape[-1]
        expanded_attr = np.repeat(attr[:, :, None, :], n_circle, axis=2)
        expanded_attr = expanded_attr.reshape(N * S * n_circle, num_channels)
        assert expanded_attr.shape[0] == verts.shape[0], \
            f"❌ verts.shape[0] = {verts.shape[0]}, expanded_attr.shape[0] = {expanded_attr.shape[0]}"

    return (verts, faces_all, expanded_attr) if expanded_attr is not None else (verts, faces_all)

def align_wisp(
    strands: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Align wisp strands to a canonical orientation.

    Args:
        strands (torch.Tensor): Input strands of shape (N, V, 3), must be on CUDA.
        beta (float): Smoothing parameter.

    Returns:
        Tuple containing:
        - aligned_strands (torch.Tensor): Aligned strands of shape (N, V, 3).
        - handle_strands (torch.Tensor): Smoothed handle strands of shape (N, V, 3).
        - longest_idx (int): Index of the longest strand.
    """
    # Ensure input is on CUDA
    if not strands.is_cuda:
        raise ValueError("Input strands must reside on CUDA.")

    # Check input shape
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("Input tensor must have shape (N, V, 3)")

    num_strands, num_points, _ = strands.shape
    dtype = strands.dtype
    device = strands.device

    # Center the strands at the mean of root positions
    root_center = strands[:, 0, :].mean(dim=0, keepdim=True)  # (1, 3)
    strands = strands - root_center

    roots = strands[:, 0, :].cpu().numpy()  # (N, 3)

    try:
        _, S, Vt = np.linalg.svd(roots, full_matrices=False)
        principal_axis_np = Vt[0] if np.all(np.isfinite(Vt[0])) else np.array([1.0, 0.0, 0.0])
    except np.linalg.LinAlgError:
        principal_axis_np = np.array([1.0, 0.0, 0.0])

    principal_axis = torch.from_numpy(principal_axis_np).to(dtype=dtype, device=device)

    # Compute strand lengths and find the longest one
    deltas = torch.norm(strands[:, 1:] - strands[:, :-1], dim=2)  # (N, V-1)
    lengths = deltas.sum(dim=1)  # (N,)
    longest_idx = torch.argmax(lengths)

    # Smooth the strands
    handle_strands = smooth_strands(strands, beta=beta)
    guide_strand = handle_strands[longest_idx]  # (V, 3)

    # Tangent vectors at the root
    t0 = guide_strand[1] - guide_strand[0]
    t1 = guide_strand[2] - guide_strand[1]

    t0 = t0 - torch.dot(t0, principal_axis) * principal_axis
    t1 = t1 - torch.dot(t1, principal_axis) * principal_axis

    # Normalize tangents
    t0 = t0 / torch.norm(t0, dim=-1, keepdim=True)
    t1 = t1 / torch.norm(t1, dim=-1, keepdim=True)

    # Binormal and normal vectors using cross product
    b0 = torch.cross(t0, t1, dim=-1)
    b0 /= torch.norm(b0, dim=-1, keepdim=True)
    n0 = torch.cross(b0, t0, dim=-1)
    n0 /= torch.norm(n0, dim=-1, keepdim=True)

    if not (torch.all(torch.isfinite(n0)) and torch.all(torch.isfinite(b0))):
        n0, b0 = tangent_frame(t0.unsqueeze(0))
        n0, b0 = n0.squeeze(0), b0.squeeze(0)

    assert torch.all(torch.isfinite(t0))
    assert torch.all(torch.isfinite(b0))
    assert torch.all(torch.isfinite(n0))

    # Define target frame
    T_dst = torch.tensor([0, -1, 0], dtype=dtype, device=device)
    N_dst = torch.tensor([0, 0, -1], dtype=dtype, device=device)
    B_dst = torch.tensor([1, 0, 0], dtype=dtype, device=device)

    # Construct rotation matrix from source to destination
    R_src = torch.stack([t0, n0, b0], dim=-1)  # (3, 3)
    R_dst = torch.stack([T_dst, N_dst, B_dst], dim=-1)   # (3, 3)
    R_global = torch.matmul(R_dst, R_src.transpose(-1, -2))  # (3, 3)

    # Rotate the entire strand set
    strands = torch.matmul(strands.view(-1, 3), R_global.T).view(num_strands, num_points, 3)
    handle_strands = torch.matmul(handle_strands.view(-1, 3), R_global.T).view(num_strands, num_points, 3)

    # Align the canonical handles and compute the local rotations
    T_dst = T_dst.unsqueeze(0)
    roots = strands[:, 0, :]

    canonical_handles = compute_aligned_strands(
        handle_strands,
        roots=roots,
        T_dst=T_dst,
    )

    R_local = compute_alignment_rotation(handle_strands, T_dst)
    aligned_strands = skinning(strands, handle_strands, canonical_handles, R_local, beta=beta)

    return aligned_strands, handle_strands, longest_idx


def align_wisps(
    wisps: List[np.ndarray],  # <- actual input is list of NumPy arrays
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align wisp strands to a canonical orientation.

    Args:
        wisps (List[np.ndarray]): List of strand arrays of shape (N_i, V, 3), one per wisp.
        beta (float): Smoothing parameter for handle strands.

    Returns:
        Tuple containing:
        - aligned_strands (torch.Tensor): Aligned strands of shape (N, V, 3).
        - handle_strands_orig (torch.Tensor): Smoothed handle strands of shape (N, V, 3).
        - longest_idx (torch.Tensor): Global indices of the longest strand per wisp.
    """
    wisps = [torch.from_numpy(np.asarray(w)) for w in wisps]

    dtype = torch.float32
    device = "cuda"
    strands = torch.cat(wisps, dim=0).to(dtype=dtype, device=device)

    # Check input shape
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("Input tensor must have shape (N, V, 3)")

    num_strands, num_points, _ = strands.shape

    deltas = torch.norm(strands[:, 1:] - strands[:, :-1], dim=2)  # (N, V-1)
    lengths = deltas.sum(dim=1)  # (N,)

    handle_strands_orig = smooth_strands(strands, beta=beta)

    # Define target frame
    T_dst = torch.tensor([0, 1, 0], dtype=dtype, device=device)
    N_dst = torch.tensor([0, 0, -1], dtype=dtype, device=device)
    B_dst = torch.tensor([-1, 0, 0], dtype=dtype, device=device)

    strands_all = []
    handle_strands_all = []
    longest_idx_all = []

    offset = 0
    for w in tqdm(wisps, desc="Aligning wisps"):
        idx_start = offset
        idx_end = offset + w.shape[0]
        offset += w.shape[0]

        strands_wisp = strands[idx_start: idx_end].clone()
        handle_strands_wisp = handle_strands_orig[idx_start: idx_end].clone()
        lengths_wisp = lengths[idx_start: idx_end]

        root_center = strands_wisp[:, 0, :].mean(dim=0, keepdim=True)  # (1, 3)
        strands_wisp -= root_center
        handle_strands_wisp -= root_center
        roots = strands_wisp[:, 0, :].cpu().numpy()

        try:
            _, _, Vt = np.linalg.svd(roots, full_matrices=False)
            if Vt.shape == (3, 3) and np.all(np.isfinite(Vt[:3])):
                pca_axes_np = Vt[:3]  # principal, second, least (3, 3)
            else:
                pca_axes_np = np.eye(3, dtype=np.float32)
        except np.linalg.LinAlgError:
            pca_axes_np = np.eye(3, dtype=np.float32)

        pca_axes = torch.from_numpy(pca_axes_np).to(dtype=dtype, device=device)
        least_axis, second_axis, principal_axis = pca_axes[2], pca_axes[1], pca_axes[0]

        longest_idx_wisp = torch.argmax(lengths_wisp).item()
        # guide_strand = handle_strands_wisp[longest_idx_wisp]  # (V, 3)
        guide_strand = strands_wisp[longest_idx_wisp]  # (V, 3)

        longest_idx_all.append(idx_start + longest_idx_wisp)

        def safe_sign(x: torch.Tensor) -> float:
            """Returns sign(x) as a scalar float. Replaces 0 with +1."""
            val = torch.sign(x).item()
            return 1.0 if val == 0.0 else val

        view = guide_strand[0] - guide_strand[1]
        view = view / torch.norm(view, dim=-1, keepdim=True).clamp(min=1e-8)

        down = guide_strand[-1] - guide_strand[0]
        down = down / torch.norm(down, dim=-1, keepdim=True).clamp(min=1e-8)

        # T_src = second_axis * safe_sign(torch.dot(second_axis, down))
        N_src = least_axis * safe_sign(torch.dot(least_axis, view))

        T_src = down - torch.dot(down, N_src) * N_src
        T_src = T_src / torch.norm(T_src, dim=-1, keepdim=True)

        if not torch.all(torch.isfinite(T_src)):
            T_src = second_axis

        B_src = torch.cross(T_src, N_src, dim=-1)

        R_src = torch.stack([T_src, N_src, B_src], dim=-1)  # (3, 3)
        R_dst = torch.stack([T_dst, N_dst, B_dst], dim=-1)  # (3, 3)
        R_global = R_dst @ R_src.transpose(-1, -2)

        # Rotate the entire strand set
        strands_wisp = torch.matmul(strands_wisp.view(-1, 3), R_global.T).view(-1, num_points, 3)
        handle_strands_wisp = torch.matmul(handle_strands_wisp.view(-1, 3), R_global.T).view(-1, num_points, 3)

        strands_all.append(strands_wisp)
        handle_strands_all.append(handle_strands_wisp)

    strands = torch.cat(strands_all, dim=0)
    handle_strands = torch.cat(handle_strands_all, dim=0)
    longest_idx = torch.tensor(longest_idx_all, dtype=torch.int64, device=device)

    # Align the canonical handles and compute the local rotations
    T_dst = T_dst.unsqueeze(0)
    roots = strands[:, 0, :]

    canonical_handles = compute_aligned_strands(
        handle_strands,
        roots=roots,
        T_dst=T_dst,
    )

    R_local = compute_alignment_rotation(handle_strands, T_dst)
    aligned_strands = skinning(strands, handle_strands, canonical_handles, R_local, beta=beta)

    return aligned_strands, handle_strands_orig, longest_idx


def statistical_outlier_mask(
    points: NDArray[np.float32],  # shape = (N, 3)
    n_neighbors: int = 5,
    ratio: float = 2.0,
) -> NDArray[np.bool_]:
    """
    Identify non-outlier points using statistical outlier filtering.

    For each point, computes the average distance to its k nearest neighbors,
    and removes points whose average distance exceeds (ratio × global mean distance).

    Args:
        points (NDArray): Input array of shape (N, 3).
        n_neighbors (int): Number of neighbors to consider for each point.
        ratio (float): Multiplier for the distance threshold.

    Returns:
        NDArray[np.bool_]: Boolean mask of shape (N,) indicating non-outlier points.
    """
    N = points.shape[0]
    if N <= 1:
        return np.ones(N, dtype=np.bool_)

    tree = KDTree(points)
    k = min(N - 1, n_neighbors)
    distances, _ = tree.query(points, k + 1)  # Includes the point itself at index 0

    # Exclude self-distance (0.0)
    neighbor_distances = distances[:, 1:]  # shape: (N, k)
    mean_neighbor_distances = np.mean(neighbor_distances, axis=1)  # (N,)

    global_mean = np.mean(mean_neighbor_distances)
    threshold = ratio * global_mean

    return mean_neighbor_distances < threshold
