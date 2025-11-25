import numpy as np
import torch
import pystrns2cards as s2c
import os
import argparse
import polyscope as ps
from PIL import Image
import re

from texture_utils import pack_wisps, render_texture, render_patches_reduced
from texture_utils import combine_wisps_with_cluster_labels, generate_uv_texture

np.random.seed(42)

NUM_SAMPLES_PER_STRAND = 100

def strand_to_strip_mesh(
    strand: np.ndarray,
    res_w: int = 2,
    width: float = 0.5,
    tangent_rotations_deg: list = [0.0]
) -> list[np.ndarray]:
    """
    Generate one or more strips by rotating the initial RMF frame and propagating.
    Returns a list of (res_w, S, 3) arrays.
    """
    assert res_w >= 2, "res_w must be at least 2"
    S = strand.shape[0]
    deltas = strand[1:] - strand[:-1]
    tangents = np.zeros_like(strand)
    tangents[:-1, :] += deltas
    tangents[1:, :] += deltas
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True).clip(min=1e-8)

    # --- Root RMF frame ---
    T0 = tangents[0]
    dT = tangents[1] - tangents[0]

    if np.linalg.norm(dT) < 1e-4:
        # Fallback: default normal orthogonal to T0
        N_base = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(T0, N_base)) > 0.9:
            N_base = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        N_base = -dT / np.linalg.norm(dT)

    B_base = np.cross(T0, N_base)
    B_base /= np.linalg.norm(B_base).clip(min=1e-8)
    N_base = np.cross(B_base, T0)  # Re-orthogonalize

    all_strips = []

    for angle_deg in tangent_rotations_deg:
        # --- Rotate initial N and B around T0 ---
        theta = np.deg2rad(angle_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Rodrigues' rotation of N_base around T0
        dot_TN = np.dot(T0, N_base)
        N0 = (
            cos_theta * N_base +
            sin_theta * np.cross(T0, N_base) +
            (1.0 - cos_theta) * dot_TN * T0
        )
        B0 = np.cross(T0, N0)

        # --- RMF propagation ---
        N = np.zeros_like(strand)
        B = np.zeros_like(strand)
        N[0] = N0
        B[0] = B0
        for i in range(1, S):
            v = strand[i] - strand[i - 1]
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-8:
                N[i] = N[i - 1]
                B[i] = B[i - 1]
                continue
            v = v / v_norm

            nL = N[i - 1] - 2 * np.dot(N[i - 1], v) * v
            tL = tangents[i - 1] - 2 * np.dot(tangents[i - 1], v) * v
            v2 = tangents[i] - tL
            v2 /= np.linalg.norm(v2).clip(min=1e-8)
            n1 = nL - 2 * np.dot(nL, v2) * v2
            n1 /= np.linalg.norm(n1).clip(min=1e-8)
            N[i] = n1
            B[i] = np.cross(tangents[i], n1)

        # --- Build strip ---
        offsets = np.linspace(-0.5 * width, 0.5 * width, res_w, dtype=np.float32)[:, None, None]
        strip = strand[None, :, :] + offsets * B[None, :, :]  # (res_w, S, 3)
        all_strips.append(strip)

    return all_strips  # list of (res_w, S, 3)

def visualize_strands(strands, name="guide strands", radius=2e-3):
    """
    Visualize a list of (S_i, 3) strands using Polyscope.
    """
    # Flatten points and build edge list
    all_points = []
    edges = []
    offset = 0

    for strand in strands:
        n_pts = strand.shape[0]
        all_points.append(strand)
        edges += [[offset + i, offset + i + 1] for i in range(n_pts - 1)]
        offset += n_pts

    all_points = np.concatenate(all_points, axis=0)
    edges = np.array(edges, dtype=np.int32)

    # Visualize
    ps.register_curve_network(name, all_points, edges, radius=radius)

def visualize_packing(packed_wisps, uv_coords, tube_radius, seed=42):
    np.random.seed(seed)

    all_strands = []
    cluster_indices = []

    for idx, (wisp_idx, _, _, _, _) in enumerate(uv_coords):
        assert wisp_idx >= 0
        packed_wisp = packed_wisps[wisp_idx]
        all_strands.append(packed_wisp)
        cluster_indices.append(np.full((packed_wisp.shape[0], packed_wisp.shape[1], 1), idx))

    all_strands = np.concatenate(all_strands, axis=0)
    cluster_indices = np.concatenate(cluster_indices, axis=0)

    verts, faces, vertex_clusters = s2c.strands_to_tube(all_strands, radius=tube_radius, n_circle=3, attr=cluster_indices)

    vertex_clusters = vertex_clusters.squeeze()
    num_clusters = int(vertex_clusters.max()) + 1

    random_colors = np.random.uniform(0.3, 1.0, size=(num_clusters, 3))
    vertex_colors = random_colors[vertex_clusters.astype(np.int32)]

    ps_mesh = ps.register_surface_mesh("PackedWisps", verts, faces, edge_width=1.0)
    ps_mesh.add_color_quantity("Cluster Colors", vertex_colors, defined_on='vertices', enabled=True)

def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-8)
    return normals

def compute_face_centers(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0

def allocate_vertex_counts(lengths: np.ndarray, total_budget: int) -> np.ndarray:
    """
    Allocate total_budget vertices to strands proportionally to their lengths,
    with a minimum of 2 vertices per strand.

    Args:
        lengths (np.ndarray): (G,) array of guide strand lengths.
        total_budget (int): Total number of vertices to allocate.

    Returns:
        np.ndarray: (G,) array of allocated vertex counts (integers).
    """
    G = lengths.shape[0]
    min_vertices = 2

    # Step 1: Reserve minimum per strand
    base_allocation = np.full(G, min_vertices, dtype=np.int32)
    remaining_budget = total_budget - min_vertices * G
    if remaining_budget < 0:
        raise ValueError("Total budget too small for minimum allocation (need at least 2 per strand).")

    # Step 2: Proportional distribution of the remaining budget
    proportions = lengths / lengths.sum()
    additional_allocation = np.floor(proportions * remaining_budget).astype(np.int32)

    # Step 3: Handle rounding error
    leftover = remaining_budget - additional_allocation.sum()
    if leftover > 0:
        # Distribute leftover to strands with largest fractional remainder
        fractional = proportions * remaining_budget - additional_allocation
        top_indices = np.argsort(-fractional)[:leftover]
        additional_allocation[top_indices] += 1

    return base_allocation + additional_allocation

def run_one_model(args, model_name):
    strands_path = os.path.join("./dataset/Groom", args.dataset, "npz", f"{model_name}.npz")

    strands = s2c.load_npz_as_strands_arc_length(strands_path, target_num_samples=NUM_SAMPLES_PER_STRAND)
    scale = np.array([1, -1, 1], dtype=np.float32)
    strands = [v * scale for v in strands]

    print(f">>> Processing: {model_name} ({len(strands)} strands)")

    experiment_ID = args.experiment_ID if args.experiment_ID is not None else "example"
    cluster_root = f"./output/{experiment_ID}"
    cluster_dirname = os.path.join(cluster_root, args.dataset, args.config, model_name)
    intermediate_dirname = os.path.join(cluster_dirname, "texture")
    output_dirname = os.path.join(cluster_dirname)

    data = np.load(os.path.join(cluster_dirname, f"{model_name}_data.npz"))
    source_indices = data["source_indices"]
    cluster_ids = data["cluster_ids"]
    guide_indices = data["guide_indices"]

    print(source_indices.shape)
    print(cluster_ids.shape)
    print(guide_indices.shape)

    strands_np = np.ascontiguousarray(np.stack(strands, axis=0)[source_indices])
    guide_strands = strands_np[guide_indices]

    print(guide_strands.shape)

    match = re.search(r"cluster(\d+)", args.config)
    if match:
        wisp_cluster_count = int(match.group(1))
        print("wisp_cluster_count =", wisp_cluster_count)
    else:
        raise ValueError(f"Cannot extract cluster count from config name: {args.config}")

    wisp_list = []
    wisp2cluster = []
    for cluster_id in range(wisp_cluster_count):
        cluster_mask = (cluster_ids == cluster_id)
        wisp = strands_np[cluster_mask]
        if wisp.size == 0:
            print(f"⚠️ WARNING: Cluster {cluster_id} is empty. Skipping.")
            continue
        wisp_list.append(wisp)
        wisp2cluster.append(cluster_id)

    aligned_strands, handle_strands_orig, longest_idx = s2c.align_wisps(wisp_list, beta=args.beta_value)

    # Split back and map to cluster_id
    lens = [len(w) for w in wisp_list]
    aligned_wisps = torch.split(aligned_strands.cpu(), lens)
    aligned_wisps = [wisp.numpy() for wisp in aligned_wisps]

    print("len(aligned_wisps) =", len(aligned_wisps))

    # === Apply Packing ===
    packed_wisps, uv_coords = pack_wisps(aligned_wisps)

    if args.n_tex_clusters is not None:
        print("Rendering reduced patches...")
        centroid_indices, labels = render_patches_reduced(
            packed_wisps,
            uv_coords,
            resolution=args.resolution//8,
            output_dirname=intermediate_dirname,
            n_clusters=args.n_tex_clusters,
            dump_patches=args.dump_patches,
        )

        # Re-pack only centroid wisps
        centroid_wisps = [packed_wisps[i] for i in centroid_indices]
        centroid_packed, centroid_uv_coords = pack_wisps(centroid_wisps)

        # Map full uv_coords through cluster labels → centroid UVs
        uv_coords = []
        for cluster_id in labels:
            idx, u0, v0, u1, v1 = centroid_uv_coords[cluster_id]
            uv_coords.append((idx, u0, v0, u1, v1))
    else:
        centroid_packed = packed_wisps
        centroid_uv_coords = uv_coords

    # === Generate Texture and UVs ===
    generate_uv_texture(
        tex_filename=os.path.join(intermediate_dirname, "packed.png"),
        layout=centroid_uv_coords,
    )

    deltas = np.diff(guide_strands, axis=1)  # shape: (G, S-1, 3)
    guide_lengths = np.linalg.norm(deltas, axis=2).sum(axis=1)     # shape: (G,)

    # Adjust vertex budget if generating multiple strips per strand
    strip_multiplier = len([-45.0, 45.0]) if (not args.disable_cross_strips) else 1
    adjusted_budget = args.num_quads // strip_multiplier + len(guide_indices)

    vertex_counts = allocate_vertex_counts(guide_lengths, adjusted_budget)

    guide_strands_resampled = []
    for strand, target_samples in zip(guide_strands, vertex_counts):
        resampled = s2c.resample_single_strand_by_arclength(strand, target_samples)
        guide_strands_resampled.append(resampled)

    guide_strands = guide_strands_resampled
    del guide_strands_resampled
    print(np.min(vertex_counts), np.sum(vertex_counts))

    wisps_mesh = []
    uv_coords_expanded = []
    wisp2cluster_expanded = []

    # Enable multiple strips if CLI flag is set
    tangent_rotations = [-45.0, 45.0] if (not args.disable_cross_strips) else [0.0]

    for i, strand in enumerate(guide_strands):
        strips = strand_to_strip_mesh(strand, res_w=args.res_w, width=args.strip_width, tangent_rotations_deg=tangent_rotations)
        wisps_mesh.extend(strips)

        # Duplicate the corresponding uv_coords entry for each strip
        uv_entry = uv_coords[i]
        uv_coords_expanded.extend([uv_entry] * len(strips))
        wisp2cluster_expanded.extend([wisp2cluster[i]] *  len(strips))

    verts, faces, uvs, face2cluster, quads, h_edges, w_edges = combine_wisps_with_cluster_labels(
        wisps=wisps_mesh,
        layout=uv_coords_expanded,
        wisp2cluster=wisp2cluster_expanded,
    )
    verts /= 100.0 # centimeter to meter

    output_filename = os.path.join(output_dirname, f"initial_strip.obj")
    s2c.save_obj(output_filename, verts, faces, uvs)

    np.savez_compressed(
        os.path.join(intermediate_dirname, f"{model_name}_strip_data.npz"),
        face2cluster=face2cluster,
        quads=quads,
        h_edges=h_edges,
        w_edges=w_edges,
    )

    if args.visualize_packing or args.visualize_mapping:
        # === Visualization in Polyscope ===
        ps.init()
        ps.set_ground_plane_mode("none")

        if args.visualize_packing:
            visualize_packing(centroid_packed, centroid_uv_coords, tube_radius=args.tube_radius)
        elif args.visualize_mapping:
            ps_mesh = ps.register_surface_mesh("AllWisps", verts, faces)
            uvs[:, 1] = 1.0 - uvs[:, 1]
            ps_mesh.add_parameterization_quantity("uv_coords", uvs, defined_on='vertices', enabled=True)

            tex_filename=os.path.join(intermediate_dirname, "packed.png")
            texture_img = np.asarray(Image.open(tex_filename).convert("RGB")).astype(np.float32) / 255.0
            ps_mesh.add_color_quantity("wisp_texture", texture_img, defined_on='texture', param_name="uv_coords", enabled=True)

        ps.show()
        return

    print("Rendering texture...")
    render_texture(
        centroid_packed,
        centroid_uv_coords,
        tube_radius=args.tube_radius,
        n_circle=args.n_circle,
        resolution=args.resolution,
        output_dirname=output_dirname,
        subsampling_ratio=args.subsampling_ratio,
        )

    if args.show:
        ps.init()
        visualize_strands(guide_strands)
        face_normals = compute_face_normals(verts, faces)
        face_centers = compute_face_centers(verts, faces)
        verts = verts * 100.0
        ps.register_surface_mesh("guide strips", verts, faces, edge_width=1.0)
        pc = ps.register_point_cloud("face centers", face_centers, radius=1e-4, enabled=False)
        pc.add_vector_quantity("normals", face_normals, enabled=True)
        all_strip_points = np.concatenate([strip.reshape(-1, 3) for strip in wisps_mesh], axis=0)
        ps.register_point_cloud("wisps_mesh points", all_strip_points, radius=0.001)
        ps.show()
        return

    log_lines = []
    log_lines.append("=== Parameters ===")
    for key, value in vars(args).items():
        log_lines.append(f"{key}: {value}")
    log_lines.append("")

    log_path = os.path.join(intermediate_dirname, "texture_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate wisp texture and initial polygon strips")
    parser.add_argument("--resolution", type=int, default=4096, help="Output texture resolution.")
    parser.add_argument("--experiment_ID", type=str, default="example", help="Dataset name")
    parser.add_argument("--dataset", type=str, default="CT2Hair", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="All", help="Model name, or 'All'")
    parser.add_argument("--config", type=str, default="cluster300", help="Name of clustering config")
    parser.add_argument("--n_tex_clusters", type=int, default=20, help="Number of texture clusters.")
    parser.add_argument("--beta_value", type=float, default=2.0, help="Smoothing parameter for strand alignment (larger -> more detail preserving).")
    parser.add_argument("--tube_radius", type=float, default=0.005, help="Tupe radius for texture rendering.")
    parser.add_argument("--n_circle", type=int, default=8, help="Number of circle samples for tube cross-section.")
    parser.add_argument("--num_quads", type=int, default=15_000, help="Number of quad faces")
    parser.add_argument("--res_w", type=int, default=2, help="Number of vertices across the strip width")
    parser.add_argument("--strip_width", type=float, default=0.1, help="Initial width of each polystrip in cm")
    parser.add_argument("--disable_cross_strips", action="store_true", help="Disable multiple strips per wisp with ±45° rotations.")
    parser.add_argument("--subsampling_ratio", type=float, default=None, help="Subsampling ratio for random strand descimation before texture rendering (default: disabled)")
    parser.add_argument("--show", action="store_true", help="Launch visualizer")
    parser.add_argument("--dump_patches", action="store_true", help="Dump texutre patches for debugging compression")
    parser.add_argument("--visualize_packing", action="store_true", help="Toggle packing visualization.")
    parser.add_argument("--visualize_mapping", action="store_true", help="Toggle texture mapping visualization.")
    args = parser.parse_args()

    if args.model_name.lower() == "all":
        npz_dir = os.path.join("./dataset/Groom", args.dataset, "npz")
        model_names = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(npz_dir)
            if f.endswith(".npz") and not f.startswith(".")
        ])
        print(f"📦 Found {len(model_names)} models: {model_names}")

        for model_name in model_names:
            print(f"\n=== Running model: {model_name} ===")
            run_one_model(args, model_name)
    else:
        run_one_model(args, args.model_name)

if __name__ == "__main__":
    main()
