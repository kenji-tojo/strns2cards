import torch
import numpy as np

from .antialias import *
from .bitset_utils import *
from .camera import *
from .cluster import *
from .cluster import *
from .curve_utils import *
from .dda import *
from .rasterize import *
from . import optimizer

from . import _core

def print_version():
    _core.print_version()

def build_edges_from_triangles(tri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a unique edge list and triangle-to-edge mapping from a triangle mesh.

    Args:
        tri (np.ndarray): Triangle face indices of shape (T, 3), int32, CPU-only.

    Returns:
        edges (np.ndarray): (E, 2) int32 array of unique edges.
        edge2tri (np.ndarray): (E, 2) int32 array where each row maps the edge to its adjacent triangles.
                               If an edge is on the boundary, the second entry will be -1.
    """
    if not isinstance(tri, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    if tri.dtype != np.int32:
        raise TypeError("Triangle indices must be int32.")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("Input triangle array must have shape (T, 3).")

    tri = np.ascontiguousarray(tri)
    edges, edge2tri = _core.build_edges_from_triangles(tri)
    return edges, edge2tri

def load_strands(filepath: str, return_raw: bool = False):
    """
    Load strands from a file into an (N,) object array of (M_i, 3) numpy arrays.

    Args:
        filepath (str): Path to the strand file.
        return_raw (bool): If True, also return raw (num_samples, points) arrays.

    Returns:
        np.ndarray: Object array where each entry is a (M_i, 3) float32 array.
        (optional) Tuple (num_samples, points): Raw internal flattened representation.
    """
    num_samples, points = _core.load_strands(filepath)
    num_samples = np.asarray(num_samples).ravel()
    points = np.asarray(points)

    strands = np.empty(len(num_samples), dtype=object)
    offset = 0
    for i, n in enumerate(num_samples):
        strand = points[offset:offset+n]
        strands[i] = strand
        offset += n

    if offset != len(points):
        raise RuntimeError("load_strands: Offset mismatch after slicing points.")

    if return_raw:
        return strands, num_samples, points
    else:
        return strands

def resample_strands_by_arclength(
    num_samples: np.ndarray,
    points: np.ndarray,
    target_num_samples: int = 100,
) -> np.ndarray:
    """
    Resample each strand to a fixed number of points based on arc-length interpolation.

    Args:
        num_samples (np.ndarray): 1D array of shape (N,) containing the number of points per strand.
        points (np.ndarray): 2D array of shape (sum(num_samples), 3) containing all strand points concatenated.
        target_num_samples (int): Desired number of resampled points per strand.

    Returns:
        np.ndarray: A (N, target_num_samples, 3) float32 array of resampled strands.
                    Each strand has exactly `target_num_samples` points evenly spaced by arc length.
    """
    num_samples = np.asarray(num_samples).ravel()
    points = np.asarray(points)
    return _core.resample_strands_by_arclength(num_samples, points, target_num_samples)

def load_strands_arc_length(filepath: str, target_num_samples: int = 100) -> np.ndarray:
    """
    Load and resample strands to fixed number of samples per strand.

    Returns:
        np.ndarray: (N, target_num_samples, 3) float32 array.
    """
    num_samples, points = _core.load_strands(filepath)
    return resample_strands_by_arclength(num_samples, points, target_num_samples)

def save_obj(filename, verts, faces, uvs):
    """
    Save mesh data as an OBJ file with UVs.

    Args:
        filename (str): Output OBJ filename.
        verts (np.ndarray): Vertex positions, shape (N, 3).
        faces (np.ndarray): Triangle indices, shape (M, 3).
        uvs (np.ndarray): UV coordinates, shape (N, 2).
    """
    assert verts.shape[0] == uvs.shape[0], "Number of vertices and UVs must match."

    with open(filename, "w") as f:
        # Write vertices
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write UVs
        for uv in uvs:
            f.write(f"vt {uv[0]} {1.0 - uv[1]}\n")

        # Write faces
        # OBJ indices are 1-based
        for face in faces:
            v1, v2, v3 = face + 1
            f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")

    print(f"✅ Saved OBJ to {filename}")

def load_npz_as_strands(npz_path):
    print(f"📦 Loading .npz from: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    num_samples = data['num_samples']
    points = data['points']
    strands = []
    offset = 0
    for n in num_samples:
        strands.append(points[offset:offset+n])
        offset += n
    if offset != len(points):
        raise ValueError("Mismatch between num_samples and points length.")

    total = len(strands)
    print(f"✅ Found {total} strands")

    return strands

def load_npz_as_strands_arc_length(npz_path, target_num_samples=100):
    print(f"📦 Loading .npz from: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    num_samples = data['num_samples']
    points = data['points']

    strands = resample_strands_by_arclength(num_samples, points, target_num_samples)

    return strands

def build_edge_indices_from_wisp(wisp) -> np.ndarray:
    N, S, _ = wisp.shape
    base_idx = np.arange(N * S).reshape(N, S)

    # Connect each point to the next along the strand
    edges = np.stack([
        base_idx[:, :-1].reshape(-1),
        base_idx[:, 1:].reshape(-1)
    ], axis=1)

    return edges  # shape: (N*(S-1), 2)

def resample_single_strand_by_arclength(strand: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Resample a single strand (N, 3) to a fixed number of points using arclength parameterization.

    Args:
        strand (np.ndarray): Input strand of shape (N, 3)
        target_samples (int): Desired number of points (≥ 2)

    Returns:
        np.ndarray: Resampled strand of shape (target_samples, 3)
    """
    if strand.shape[0] < 2:
        raise ValueError("Strand must have at least 2 points")

    # Step 1: Compute cumulative arclength
    deltas = np.diff(strand, axis=0)  # (N-1, 3)
    segment_lengths = np.linalg.norm(deltas, axis=1)  # (N-1,)
    arclength = np.concatenate([[0], np.cumsum(segment_lengths)])  # (N,)

    # Step 2: Create target arclengths to sample at
    target_distances = np.linspace(0, arclength[-1], target_samples)  # (target_samples,)

    # Step 3: Interpolate each coordinate dimension independently
    resampled = np.empty((target_samples, 3), dtype=np.float32)
    for d in range(3):  # x, y, z
        resampled[:, d] = np.interp(target_distances, arclength, strand[:, d])

    return resampled
