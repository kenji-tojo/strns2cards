import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
import pystrns2cards as s2c
import os
import argparse
import polyscope as ps
from PIL import Image
from tqdm import tqdm 
import trimesh
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def build_clustered_strand_edges(strands, strand_cluster_ids):
    """
    Build edge geometry for visualization of strands, associating each edge with a cluster ID.

    Args:
        strands (List[np.ndarray]): List of strands, each with shape (num_points, 3).
        strand_cluster_ids (List[int]): Cluster ID associated with each strand.

    Returns:
        points (np.ndarray): Concatenated points of all strands, shape (total_points, 3).
        edges (np.ndarray): Edge indices connecting consecutive points, shape (total_edges, 2).
        edge_cluster_ids (np.ndarray): Cluster ID per edge, shape (total_edges,).
        num_clusters (int): Number of unique clusters.
    """
    assert len(strands) == len(strand_cluster_ids), "Mismatch between strands and cluster IDs"

    points = []
    edges = []
    edge_cluster_ids = []

    point_offset = 0
    num_clusters = 0

    for strand, cluster_id in zip(strands, strand_cluster_ids):
        num_points = strand.shape[0]
        
        points.append(strand)
        
        strand_edges = [[point_offset + i, point_offset + i + 1] for i in range(num_points - 1)]
        edges.extend(strand_edges)
        
        strand_edge_clusters = [cluster_id] * (num_points - 1)
        edge_cluster_ids.extend(strand_edge_clusters)

        num_clusters = max(num_clusters, cluster_id + 1)
        point_offset += num_points

    points = np.concatenate(points, axis=0)
    edges = np.array(edges, dtype=np.int32)
    edge_cluster_ids = np.array(edge_cluster_ids, dtype=np.int32)

    assert len(edges) == len(edge_cluster_ids), "Mismatch between edges and edge cluster IDs"

    return points, edges, edge_cluster_ids, num_clusters

def build_ortho_cameras(points, margin=0.2):
    """
    Build orthographic camera matrices explicitly from bbox corners, face centers, and edge centers.
    Each view uses uniform scaling in X/Y to fit points within [-(1-margin), (1-margin)], 
    and independently maps the Z axis to [-0.5, 0.5].

    Args:
        points (torch.Tensor): (N,3) points tensor.
        margin (float): Margin in XY around points in NDC space (default 0.1).

    Returns:
        MVP (torch.Tensor): (13,4,4) orthographic camera matrices tensor.
    """
    assert 0 <= margin < 1, "Margin must be in [0, 1)."

    min_xyz, _ = points.min(dim=0)
    max_xyz, _ = points.max(dim=0)
    center = (min_xyz + max_xyz) / 2
    bbox_extent = max_xyz - min_xyz
    half_extent = bbox_extent / 2
    ex, ey, ez = half_extent

    view_points = torch.stack([
        # 4 diagonal corners
        torch.tensor([ ex,  ey,  ez], device=points.device),
        torch.tensor([-ex,  ey,  ez], device=points.device),
        torch.tensor([ ex, -ey,  ez], device=points.device),
        torch.tensor([ ex,  ey, -ez], device=points.device),

        # 3 face centers
        torch.tensor([ ex, 0.0, 0.0], device=points.device),
        torch.tensor([0.0,  ey, 0.0], device=points.device),
        torch.tensor([0.0, 0.0,  ez], device=points.device),

        # 6 edge centers
        torch.tensor([ ex,  ey, 0.0], device=points.device),
        torch.tensor([ ex, -ey, 0.0], device=points.device),
        torch.tensor([ ex, 0.0,  ez], device=points.device),
        torch.tensor([ ex, 0.0, -ez], device=points.device),
        torch.tensor([0.0,  ey,  ez], device=points.device),
        torch.tensor([0.0,  ey, -ez], device=points.device),
    ], dim=0)  # 13 views

    MVP = []

    for vp in view_points:
        direction = -vp / torch.norm(vp)

        # Stable up-vector selection
        up = torch.tensor([0., 1., 0.], device=points.device)
        if torch.abs(torch.dot(direction, up)) > 0.9:
            up = torch.tensor([1., 0., 0.], device=points.device)

        right = torch.cross(up, direction, dim=-1)
        right /= torch.norm(right)
        up = torch.linalg.cross(direction, right)

        R = torch.stack([right, up, direction], dim=1)

        view_mat = torch.eye(4, device=points.device)
        view_mat[:3, :3] = R.T
        view_mat[:3, 3] = -R.T @ center

        # Points in camera space
        points_cam = (points - center) @ R
        min_cam, _ = points_cam.min(dim=0)
        max_cam, _ = points_cam.max(dim=0)
        extent_cam = max_cam - min_cam

        # Uniform scaling in XY with margin
        scale_xy = (2.0 * (1.0 - margin)) / torch.max(extent_cam[:2])

        # Independent scaling for Z to [-0.5, 0.5]
        scale_z = 1.0 / extent_cam[2]

        # Centering offset after scaling
        offset_xy = -(min_cam[:2] + max_cam[:2]) / 2 * scale_xy
        offset_z = -(min_cam[2] + max_cam[2]) / 2 * scale_z

        proj_mat = torch.eye(4, device=points.device)
        proj_mat[0, 0] = scale_xy
        proj_mat[1, 1] = scale_xy
        proj_mat[2, 2] = scale_z
        proj_mat[:2, 3] = offset_xy
        proj_mat[2, 3] = offset_z

        MVP_mat = proj_mat @ view_mat
        MVP.append(MVP_mat)

    MVP = torch.stack(MVP, dim=0)
    return MVP

def render_bitset_from_triangles(
    resolution, pos_ndc_h, tri, tri2cluster, cluster_count,
):
    H, W = resolution

    frag_pix, frag_attr = s2c.compute_fragments((H, W), pos_ndc_h, tri)
    bitset = s2c.cluster_mask_from_fragments(
        (H, W), frag_pix, frag_attr, tri2cluster, cluster_count=cluster_count,
    )
    return bitset

def colormap_bitset(
    bitset, cluster2rgb,
):
    accum = s2c.accumulate_bitset_rgb(bitset, cluster2rgb)
    pop = s2c.popcount_bitset(bitset).clamp(min=1.0).unsqueeze(-1)
    rgb = (accum / pop).clamp(0, 1).cpu().numpy()
    return rgb

def save_imgs(out_dirname, file_prefix, rgb):
    for view_idx in range(rgb.shape[0]):
        img = rgb[view_idx]  # select view explicitly, shape (H,W,3)
        img = (img * 255).astype(np.uint8)
        img_path = os.path.join(out_dirname, f"{file_prefix}_{view_idx:02d}.png")
        Image.fromarray(img).save(img_path)
        # print(f"✅ Saved image to {img_path}")

def planarity_loss(pos: torch.Tensor, quads: torch.Tensor) -> torch.Tensor:
    """
    Planarity loss for quad strips.

    Args:
        pos: (V, 3) tensor of vertex positions.
        quads: (M, 4) tensor of quad indices in CCW order.

    Returns:
        Scalar loss.
    """
    q1 = pos[quads[:, 0]]
    q2 = pos[quads[:, 1]]
    q3 = pos[quads[:, 2]]
    q4 = pos[quads[:, 3]]

    diag_diff = (q1 + q3) - (q2 + q4)  # (M, 3)
    loss = (diag_diff ** 2).sum()  # scalar
    return loss

def run_one_model(args, model_name):
    strands_path = os.path.join("./dataset/Groom", args.dataset, "npz", f"{model_name}.npz")

    strands = s2c.load_npz_as_strands(strands_path)
    scale = np.array([1, -1, 1], dtype=np.float32)
    strands = [v * scale for v in strands]

    print(f">>> Processing: {model_name} ({len(strands)} strands)")

    experiment_ID = args.experiment_ID if args.experiment_ID is not None else "test_experiment"
    cluster_root = f"./output/{experiment_ID}"
    cluster_dirname = os.path.join(cluster_root, args.dataset, args.config, model_name)
    intermediate_dirname = os.path.join(cluster_dirname, "fitting")
    imgs_dirname = os.path.join(intermediate_dirname, "imgs")
    output_dirname = os.path.join(cluster_dirname)

    os.makedirs(intermediate_dirname, exist_ok=True)
    os.makedirs(imgs_dirname, exist_ok=True)

    data = np.load(os.path.join(cluster_dirname, f"{model_name}_data.npz"))
    source_indices = data["source_indices"]
    cluster_ids = data["cluster_ids"]

    print(source_indices.shape)
    print(cluster_ids.shape)

    strands = [strands[idx] for idx in source_indices]
    points, edges, edge2cluster, cluster_count = build_clustered_strand_edges(strands, cluster_ids)

    cluster_count = cluster_count.item(0)

    points = torch.from_numpy(points).to(dtype=torch.float32, device="cuda")
    edges = torch.from_numpy(edges).to(dtype=torch.int32, device="cuda")
    edge2cluster = torch.from_numpy(edge2cluster).to(dtype=torch.int32, device="cuda")

    H = W = args.resolution

    # Assume build_ortho_cameras is your provided function
    MVP = build_ortho_cameras(points, margin=args.margin)  # (13,4,4), from your existing function

    # Homogeneous coordinates of points
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)[None, ...]  # (N,4)
    points_ndc_h = torch.matmul(points_h, MVP.transpose(1, 2))

    # Apply the first MVP (or loop through all MVPs to test each)
    for view_idx in range(MVP.shape[0]):
        points_ndc = points_ndc_h[view_idx, :, :3] / points_ndc_h[view_idx, :, [3]]
        min_ndc, _ = points_ndc.min(dim=0)
        max_ndc, _ = points_ndc.max(dim=0)
        print(f"View {view_idx}: NDC bounding box min={min_ndc.cpu().numpy()}, max={max_ndc.cpu().numpy()}")

    print("cluster_count =", cluster_count)

    frag_pix, frag_attrs = s2c.dda_compute_fragments((H, W), points_ndc_h, edges)
    target_bitset = s2c.cluster_mask_from_fragments(
        resolution=(H, W),
        frag_pix=frag_pix,
        frag_attrs=frag_attrs,
        tri2cluster=edge2cluster,
        cluster_count=cluster_count,
    )

    cluster2rgb = torch.rand(cluster_count, 3, dtype=torch.float32, device="cuda")
    rgb = colormap_bitset(target_bitset, cluster2rgb)

    print(target_bitset.shape)
    print(rgb.shape) # (B, H, W, 3), where B = 13

    save_imgs(imgs_dirname, "target", rgb)

    del edges
    del edge2cluster


    obj_path = os.path.join(output_dirname, "initial_strip.obj")
    mesh = trimesh.load(obj_path, process=False)

    assert mesh.visual.uv is not None, "OBJ file does not contain UV coordinates."

    pos_np =  mesh.vertices
    tri_np = mesh.faces.astype(np.int32)
    uv_np = mesh.visual.uv

    print(f"Loaded OBJ: pos.shape={pos_np.shape}, tri.shape={tri_np.shape}, uv.shape={uv_np.shape}")

    edges_np, edge2tri_np = s2c.build_edges_from_triangles(tri_np)

    pos = torch.from_numpy(pos_np).to(dtype=torch.float32, device="cuda")
    pos *= 100.0  # Convert from meter to centimeter
    tri = torch.from_numpy(tri_np).to(dtype=torch.int32, device="cuda")
    edges = torch.from_numpy(edges_np).to(dtype=torch.int32, device="cuda")
    edge2tri = torch.from_numpy(edge2tri_np).to(dtype=torch.int32, device="cuda")

    strip_data = np.load(os.path.join(output_dirname, "texture", f"{model_name}_strip_data.npz"))
    tri2cluster = strip_data["face2cluster"]
    quads = strip_data["quads"]
    h_edges = strip_data["h_edges"]
    w_edges = strip_data["w_edges"]


    if args.show_initial:
        pos = pos.cpu().numpy()
        tri_faces = np.vstack([
            quads[:, [0, 1, 2]],
            quads[:, [0, 2, 3]]
        ])
        ps.init()
        ps.register_surface_mesh("quads_as_triangles", pos, tri_faces, transparency=0.5)
        ps.register_curve_network("h_edges", pos, h_edges, radius=1e-3)
        ps.register_curve_network("w_edges", pos, w_edges, radius=1e-3)
        ps.show()
        return


    tri2cluster = torch.from_numpy(tri2cluster).to(dtype=torch.int32, device="cuda")
    quads = torch.from_numpy(quads).to(dtype=torch.int32, device="cuda")
    h_edges = torch.from_numpy(h_edges).to(dtype=torch.int32, device="cuda")
    w_edges = torch.from_numpy(w_edges).to(dtype=torch.int32, device="cuda")
    quad_edges = torch.cat([h_edges, w_edges], dim=0)
    edge2cluster = tri2cluster[edge2tri[:, 0]].contiguous()

    assert tri2cluster.max().item() < cluster_count
    assert torch.all(edge2cluster >= 0)

    print(tri2cluster.shape)
    print(edge2cluster.shape)
    print(quads.shape)
    print(h_edges.shape)
    print(w_edges.shape)

    pos_h = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    pos_ndc_h = torch.matmul(pos_h, MVP.transpose(1, 2))

    bitset = render_bitset_from_triangles(
        (H, W), pos_ndc_h, tri, tri2cluster, cluster_count,
    )
    print(bitset.shape)

    rgb = colormap_bitset(bitset, cluster2rgb)
    save_imgs(imgs_dirname, "initial", rgb)


    os.makedirs(os.path.join(imgs_dirname, "tmp"), exist_ok=True)


    # Large steps
    L_h = s2c.optimizer.laplacian_edges(pos, h_edges)
    L_w = s2c.optimizer.laplacian_edges(pos, w_edges)
    L = L_h + args.laplacian_w_weight * L_w

    idx = torch.arange(pos.shape[0], dtype=torch.long, device='cuda')
    eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(pos.shape[0], dtype=torch.float32, device='cuda'), (pos.shape[0], pos.shape[0]))
    lambda_ = args.laplacian_lambda
    M = torch.add(eye, lambda_ * L) # M = I + lambda_ * L
    M = M.coalesce()


    # For loss function
    L_uniform = s2c.optimizer.laplacian_edges(pos, quad_edges)


    from largesteps.parameterize import to_differential, from_differential

    # Compute the system matrix
    param_pos = to_differential(M, pos)
    param_pos.requires_grad_()

    # Optimization
    optimizer = s2c.optimizer.UniformAdam( # Uniform Adam was better than VectorAdam.
        params=[param_pos],
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    # Scheduler: decay to 1e-2 of initial lr over args.num_iters steps
    gamma = 1e-2 ** (1 / args.num_iters)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    iteration_losses = []

    for iter in tqdm(range(args.num_iters), desc="Fitting strips"):
        optimizer.zero_grad()

        pos = from_differential(M, param_pos)
        pos_h = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
        pos_ndc_h = torch.matmul(pos_h, MVP.transpose(1, 2))

        bitset = render_bitset_from_triangles(
            (H, W), pos_ndc_h, tri, tri2cluster, cluster_count,
        )

        loss = s2c.antialiased_cluster_bitset_loss(
            bitset, target_bitset, pos_ndc_h, tri,
            edges, edge2tri, edge2cluster,
            kernel_radius=7.0,
            rho=args.alpha,
            count_xor_error=args.plot_loss,
        )

        loss = loss + args.w_pln * planarity_loss(pos, quads)
        loss = loss + args.w_lap * torch.sum((L_uniform @ pos) ** 2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        iteration_losses.append(loss.item())

        # Save intermediate frame for animation
        if args.animation_view_idx is not None and iter % 3 == 0:
            rgb_anim = colormap_bitset(bitset, cluster2rgb)
            img = rgb_anim[args.animation_view_idx]  # (H, W, 3)
            img = (img * 255).astype(np.uint8)
            frame_path = os.path.join(imgs_dirname, "tmp", f"{iter:04d}.png")
            Image.fromarray(img).save(frame_path)

    rgb = colormap_bitset(bitset, cluster2rgb)
    save_imgs(imgs_dirname, "final", rgb)

    if args.animation_view_idx is not None:
        import imageio
        tmp_dir = os.path.join(imgs_dirname, "tmp")
        frames = sorted(os.listdir(tmp_dir))
        frame_paths = [os.path.join(tmp_dir, f) for f in frames if f.endswith(".png")]
        images = [imageio.imread(f) for f in frame_paths]
        video_path = os.path.join(intermediate_dirname, "animation.mp4")
        imageio.mimsave(video_path, images, fps=10)
        print(f"🎞️ Animation saved to {video_path}")

    if args.plot_loss:
        # Save and plot the loss curve
        plt.figure(figsize=(8, 4))
        plt.plot(range(args.num_iters), iteration_losses, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Optimization Loss Curve')
        plt.grid(True)
        plt.legend()

        loss_plot_path = os.path.join(intermediate_dirname, "loss_curve.png")
        plt.savefig(loss_plot_path)
        plt.close()

        print(f"📈 Loss curve plot saved to {loss_plot_path}")

    log_lines = []
    log_lines.append("=== Parameters ===")
    for key, value in vars(args).items():
        log_lines.append(f"{key}: {value}")
    log_lines.append("")

    log_path = os.path.join(intermediate_dirname, "fitting_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")


    pos = from_differential(M, param_pos)
    pos_np = pos.detach().cpu().numpy()
    pos_np /= 100.0 # centimeter to meter
    tri_np = tri.cpu().numpy()
    # uv_np is unmodified

    def _write_obj(filename, vertices, faces, uvs):
        """
        Write a Wavefront OBJ file with positions, UVs, and triangles — without .mtl reference.

        Args:
            filename (str): Output file path.
            vertices (np.ndarray): (N, 3) array of vertex positions.
            faces (np.ndarray): (M, 3) array of triangle indices.
            uvs (np.ndarray): (N, 2) array of UV coordinates (must match vertices).
        """
        assert vertices.shape[0] == uvs.shape[0], "UVs must be per-vertex and match vertex count."
        assert faces.shape[1] == 3, "Only triangular faces are supported."

        with open(filename, "w") as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for vt in uvs:
                f.write(f"vt {vt[0]} {vt[1]}\n")
            for face in faces + 1:  # OBJ format is 1-based
                f.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")

    optimized_obj_path = os.path.join(output_dirname, "strip.obj")
    _write_obj(optimized_obj_path, pos_np, tri_np, uv_np)
    print(f"💾 Optimized mesh saved to {optimized_obj_path}")

    # Explicit cycle consistency check
    original_mesh = trimesh.load(os.path.join(output_dirname, "initial_strip.obj"), process=False)
    reloaded_mesh = trimesh.load(optimized_obj_path, process=False)

    assert original_mesh.vertices.shape == reloaded_mesh.vertices.shape, "Vertex shapes do not match."
    assert np.array_equal(original_mesh.faces, reloaded_mesh.faces), "Triangle indices do not exactly match."
    assert np.allclose(original_mesh.visual.uv, reloaded_mesh.visual.uv, atol=1e-8), "UV coordinates do not exactly match."

    print("✅ UV and geometry cycle consistency verified successfully.")

    face_color = cluster2rgb.cpu().numpy()[tri2cluster.cpu().numpy()]

    if args.show_final:
        ps.init()
        mesh = ps.register_surface_mesh("optimized", pos_np, tri_np)
        mesh.add_color_quantity("cluster", face_color, defined_on="faces", enabled=True)
        ps.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize clustered hair strands.")
    parser.add_argument("--resolution", type=int, default=200, help="Output image resolution.")
    parser.add_argument("--num_iters", type=int, default=1000, help="Number of iterations.")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Negative slope of leaky-ReLU")
    parser.add_argument("--laplacian_lambda", type=float, default=100.0, help="Lambda for laplacian preconditioning.")
    parser.add_argument("--laplacian_w_weight", type=float, default=1/100, help="Weakening ratio of laplacian edge weights for w_edges")
    parser.add_argument("--w_pln", type=float, default=10.0, help="Planarity loss weight.")
    parser.add_argument("--w_lap", type=float, default=1e-2, help="Bi-Laplacian loss weight.")
    parser.add_argument("--experiment_ID", type=str, default="example", help="Dataset name")
    parser.add_argument("--dataset", type=str, default="CT2Hair", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="All", help="Model name, or 'All'")
    parser.add_argument("--config", type=str, default="cluster300", help="Name of clustering config")
    parser.add_argument("--margin", type=float, default=0.3, help="Margin of the rendering frame")
    parser.add_argument("--animation_view_idx", type=int, default=None, help="View index to save animation from")
    parser.add_argument("--plot_loss", action="store_true", help="Plot loss values")
    parser.add_argument("--show_initial", action="store_true", help="Launch visualizer to inspect inifial shape")
    parser.add_argument("--show_final", action="store_true", help="Launch visualizer to inspect final shape")
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
