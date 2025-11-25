import numpy as np
import os, shutil
import pystrns2cards as s2c

def pack_wisps(wisps):
    """
    Pack wisps into a square container without scaling and compute UV coordinates based on normalized bounding box.

    Args:
        wisps (list of np.ndarray): List of (N, V, 3) wisps.

    Returns:
        packed_wisps (list of np.ndarray): Packed wisps in their original sizes.
        uv_coords (list of tuple): UV coordinates as (idx, u0, v0, u1, v1) for each wisp.
    """
    # Compute bounding boxes and store widths/heights
    bbox_data = []
    for wisp in wisps:
        points = wisp.reshape(-1, 3)
        bbox_min = points[:, :2].min(axis=0)
        bbox_max = points[:, :2].max(axis=0)
        width, height = bbox_max - bbox_min
        bbox_data.append((bbox_min, bbox_max, width, height))

    # Extract widths and heights
    widths = np.array([bbox[2] for bbox in bbox_data])
    heights = np.array([bbox[3] for bbox in bbox_data])

    # Compute average width and height
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    # Compute total area and square size
    total_area = np.sum(widths * heights)
    square_size = np.sqrt(total_area)

    # Determine number of columns and rows
    n_rows = max(1, int(np.ceil(square_size / avg_height)))
    target_row_width = np.sum(widths) / n_rows
    print(f"Computed n_rows: {n_rows}")

    # Sort wisps by height (descending) to minimize wasted vertical space
    sorted_indices = np.argsort(-heights)

    # Initialize packing variables
    packed_wisps = [None] * len(wisps) # Prealocate based on original order
    uv_coords = [None] * len(wisps)  # Preallocate based on original order

    h_cursor = 0.0
    wisp_idx = 0 # sorted index

    for row_idx in range(n_rows):
        w_cursor = 0.0

        if wisp_idx >= len(wisps): # FIXME
            continue

        # Determine max row height (first wisp in each row due to sorting)
        max_row_height = heights[sorted_indices[wisp_idx]]

        while wisp_idx < len(wisps):
            if row_idx < n_rows - 1 and w_cursor > target_row_width:
                break

            # Access sorted wisp and bounding box data
            orig_idx = sorted_indices[wisp_idx]  # Original index in the unsorted list
            wisp = wisps[orig_idx]
            bbox_min, bbox_max, width, height = bbox_data[orig_idx]

            # UV coordinates based on current cursor positions
            u0, u1 = w_cursor, w_cursor + width
            v0, v1 = h_cursor, h_cursor + height

            # Assign UV coordinates to the original index
            uv_coords[orig_idx] = (wisp_idx, u0, v0, u1, v1)

            # Translate wisp to packed position
            offset_x = w_cursor - bbox_min[0]
            offset_y = h_cursor - bbox_min[1]
            packed_wisp = wisp.copy()
            packed_wisp[:, :, 0] += offset_x
            packed_wisp[:, :, 1] += offset_y
            packed_wisps[orig_idx] = packed_wisp

            # Update horizontal cursor
            w_cursor += width
            wisp_idx += 1

        # Move to the next row
        h_cursor += max_row_height

    # Compute the bounding square size
    all_points = np.concatenate([w.reshape(-1, 3) for w in packed_wisps], axis=0)
    bbox_min = all_points[:, :2].min(axis=0)
    bbox_max = all_points[:, :2].max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Normalize UV coordinates to [0, 1]
    uv_coords = [
        (idx, u0 / bbox_size[0], v0 / bbox_size[1], u1 / bbox_size[0], v1 / bbox_size[1])
        for idx, u0, v0, u1, v1 in uv_coords
    ]

    return packed_wisps, uv_coords

def render_texture(
    packed_wisps,
    uv_coords,
    tube_radius,
    n_circle,
    resolution=1024,
    output_dirname="./output",
    subsampling_ratio = None,
):
    import torch
    import kornia
    import numpy as np
    import imageio.v3 as iio
    from tqdm import tqdm

    # === 1. Compute patch_res from UV layout
    max_u = max(u1 - u0 for _, u0, _, u1, _ in uv_coords)
    max_v = max(v1 - v0 for _, _, v0, _, v1 in uv_coords)
    patch_res_h = int(np.ceil(max_v * resolution))
    patch_res_w = int(np.ceil(max_u * resolution))
    print(f"[INFO] Shared patch resolution: {patch_res_h}×{patch_res_w}")

    # === 2. Global world-to-ndc scaling
    packed_points = np.concatenate([w.reshape(-1, 3) for w in packed_wisps], axis=0)
    packed_bbox_min = packed_points[:, :2].min(axis=0)
    packed_bbox_max = packed_points[:, :2].max(axis=0)
    packed_bbox_size = packed_bbox_max - packed_bbox_min

    bbox_size = np.zeros(3)
    for wisp in packed_wisps:
        points = wisp.reshape(-1, 3)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = np.maximum(bbox_size, bbox_max - bbox_min)
    to_ndc_scale = 2.0 * np.ones(3) / bbox_size
    to_ndc_scale[2] *= 0.2 # not to depth-clip the strands

    print(packed_bbox_size)
    print(to_ndc_scale)

    # Initialize output atlas
    rgb_atlas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    alpha_atlas = np.zeros((resolution, resolution), dtype=np.uint8)

    for wisp, (_, u0, v0, u1, v1) in tqdm(list(zip(packed_wisps, uv_coords)), desc="Rendering wisps"):
        # randomly pick a subset of strands within wisp

        if subsampling_ratio is not None and subsampling_ratio < 1.0:
            S = wisp.shape[0]
            selected_indices = np.random.choice(S, size=max(1, int(S * subsampling_ratio)), replace=False)
            wisp = wisp[selected_indices]

        # === Compute tangents
        tangents = np.zeros_like(wisp)
        deltas = np.diff(wisp, axis=1)
        tangents[:, :-1, :] += deltas
        tangents[:, 1:, :] += deltas
        tangents /= np.linalg.norm(tangents, axis=2, keepdims=True)

        # === Generate mesh
        verts, faces, attr = s2c.strands_to_tube(
            wisp, radius=tube_radius, n_circle=n_circle, attr=tangents
        )

        u0_world = u0 * packed_bbox_size[0]
        u1_world = u1 * packed_bbox_size[0]
        v0_world = v0 * packed_bbox_size[1]
        v1_world = v1 * packed_bbox_size[1]

        verts[:, 0] -= 0.5 * (u0_world + u1_world)
        verts[:, 1] -= v0_world
        verts[:, 2] *= 1.0
        verts *= to_ndc_scale
        verts[:, 1] -= 1.0

        # === Move to GPU
        pos = torch.from_numpy(verts).to(dtype=torch.float32, device="cuda")
        tri = torch.from_numpy(faces).to(dtype=torch.int32, device="cuda")
        attr = torch.from_numpy(attr).to(dtype=torch.float32, device="cuda")

        def mitchell_netravali_filter(x, B=1/3, C=1/3):
            abs_x = np.abs(x)
            x2 = abs_x ** 2
            x3 = abs_x ** 3
            condition1 = (abs_x < 1)
            condition2 = ((abs_x >= 1) & (abs_x < 2))

            f = (
                ((12 - 9*B - 6*C) * x3 + (-18 + 12*B + 6*C) * x2 + (6 - 2*B)) * condition1 +
                ((-B - 6*C) * x3 + (6*B + 30*C) * x2 + (-12*B - 48*C) * abs_x + (8*B + 24*C)) * condition2
            ) / 6.0

            return f

        # === Rasterize into per-wisp canvas ===
        # Software MSAA
        grid_res = 8
        sampling_span = 2.0

        grid_1d = sampling_span * ((np.arange(grid_res) + 0.5) / grid_res - 0.5)
        offsets = [(dx, dy) for dy in grid_1d for dx in grid_1d]

        weights = np.array([
            mitchell_netravali_filter(dx) * mitchell_netravali_filter(dy)
            for dx, dy in offsets
        ])

        weights /= weights.sum()  # normalize
        weights = weights.tolist()
        num_samples = len(offsets)

        acc_rgb = torch.zeros((patch_res_h, patch_res_w, 3), device="cuda")
        valid_count = torch.zeros((patch_res_h, patch_res_w), device="cuda")
        acc_alpha = 0

        for i, (dx, dy) in enumerate(offsets):
            ndc_dx = dx * 2.0 / patch_res_w
            ndc_dy = dy * 2.0 / patch_res_h

            pos_jittered = pos.clone()
            pos_jittered[:, 0] += ndc_dx
            pos_jittered[:, 1] += ndc_dy
            pos_h_jittered = torch.cat([pos_jittered, torch.ones_like(pos[:, :1])], dim=-1)

            rast_out = s2c.rasterize((patch_res_h, patch_res_w), pos_h_jittered[None], tri)
            rgb = s2c.interpolate_triangle_attributes(rast_out, attr, tri)[0]
            valid = rast_out[0, ..., -1] > 0

            alpha = valid.float()

            # Accumulate RGB and valid count only for valid samples
            acc_rgb += (rgb * alpha[..., None])
            valid_count += alpha
            acc_alpha += weights[i] * alpha

        # After loop: compute average RGB from valid samples
        # Avoid division by zero by setting count to 1 temporarily
        valid_mask = valid_count > 0
        rgb = acc_rgb / valid_count.clamp(min=1)[..., None]
        rgb[~valid_mask] = torch.tensor([[0, 1, 0]], dtype=torch.float32, device="cuda")
        rgb = 0.5 * (rgb + 1.0)

        alpha = acc_alpha # weights are already normalized
        alpha = kornia.filters.gaussian_blur2d(
            alpha[None, None],
            kernel_size=(3, 3),
            sigma=(1.5, 1.5)
        )[0, 0]

        rgb = torch.flip(rgb, dims=(0,))
        alpha = torch.flip(alpha, dims=(0,))
        rgb = rgb.cpu().clamp(0, 1).numpy()
        alpha = alpha.cpu().clamp(0, 1).numpy()

        rgb_u8 = (rgb * 255).astype(np.uint8)
        alpha_u8 = (alpha * 255).astype(np.uint8)

        atlas_x0 = int(np.floor(u0 * resolution))
        atlas_y0 = int(np.floor(v0 * resolution))
        atlas_x1 = int(np.floor(u1 * resolution))
        atlas_y1 = int(np.floor(v1 * resolution))

        # 3. Convert to pixel coordinates in patch
        crop_x0 = int((patch_res_w - (atlas_x1 - atlas_x0)) / 2)
        crop_y0 = 0
        crop_x1 = crop_x0 + (atlas_x1 - atlas_x0)
        crop_y1 = crop_y0 + (atlas_y1 - atlas_y0)

        rgb_crop = rgb_u8[crop_y0:crop_y1, crop_x0:crop_x1]
        alpha_crop = alpha_u8[crop_y0:crop_y1, crop_x0:crop_x1]

        rgb_atlas[atlas_y0:atlas_y1, atlas_x0:atlas_x1] = rgb_crop
        alpha_atlas[atlas_y0:atlas_y1, atlas_x0:atlas_x1] = alpha_crop

        torch.cuda.empty_cache()

    # === Save final atlas images
    iio.imwrite(os.path.join(output_dirname, "tangent.png"), rgb_atlas)
    iio.imwrite(os.path.join(output_dirname, "tangent.tga"), rgb_atlas)
    iio.imwrite(os.path.join(output_dirname, "alpha.png"), alpha_atlas)
    iio.imwrite(os.path.join(output_dirname, "alpha.tga"), alpha_atlas)

    print(f"[✅] Texture rendering complete.")

def render_patches_reduced(
    packed_wisps,
    uv_coords,
    resolution,
    n_clusters = 20,
    output_dirname="./output",
    dump_patches=False,
):
    import torch
    import numpy as np
    import imageio.v3 as iio
    from tqdm import tqdm
    from sklearn.cluster import KMeans

    # === 1. Compute patch_res from UV layout
    max_u = max(u1 - u0 for _, u0, _, u1, _ in uv_coords)
    max_v = max(v1 - v0 for _, _, v0, _, v1 in uv_coords)
    patch_res_h = int(np.ceil(max_v * resolution))
    patch_res_w = int(np.ceil(max_u * resolution))
    print(f"[INFO] Shared patch resolution: {patch_res_h}×{patch_res_w}")

    # === 2. Global world-to-ndc scaling
    packed_points = np.concatenate([w.reshape(-1, 3) for w in packed_wisps], axis=0)
    packed_bbox_min = packed_points[:, :2].min(axis=0)
    packed_bbox_max = packed_points[:, :2].max(axis=0)
    packed_bbox_size = packed_bbox_max - packed_bbox_min

    bbox_size = np.zeros(3)
    for wisp in packed_wisps:
        points = wisp.reshape(-1, 3)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = np.maximum(bbox_size, bbox_max - bbox_min)
    to_ndc_scale = 2.0 * np.ones(3) / bbox_size
    to_ndc_scale[2] *= 0.2 # not to depth-clip the strands

    print(packed_bbox_size)
    print(to_ndc_scale)

    mask_list = []

    for wisp, (_, u0, v0, u1, v1) in tqdm(list(zip(packed_wisps, uv_coords)), desc="Rendering wisps"):
        edges = s2c.build_edge_indices_from_wisp(wisp)

        u0_world = u0 * packed_bbox_size[0]
        u1_world = u1 * packed_bbox_size[0]
        v0_world = v0 * packed_bbox_size[1]
        v1_world = v1 * packed_bbox_size[1]

        verts = wisp.copy().reshape(-1, 3)
        verts[:, 0] -= 0.5 * (u0_world + u1_world)
        verts[:, 1] -= v0_world
        verts[:, 2] *= 1.0
        verts *= to_ndc_scale
        verts[:, 1] -= 1.0

        # === Move to GPU
        pos = torch.from_numpy(verts).to(dtype=torch.float32, device="cuda")
        edges = torch.from_numpy(edges).to(dtype=torch.int32, device="cuda")

        pos_h = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=-1)

        # === Rasterize into per-wisp canvas
        frag_pix, frag_attrs = s2c.dda_compute_fragments((patch_res_h, patch_res_w), pos_h[None], edges)
        rast_out = s2c.depth_test(1, (patch_res_h, patch_res_w), frag_pix, frag_attrs)
        mask = s2c.dda_interpolate_attributes(rast_out, torch.ones_like(pos[..., :1]), edges)[0]
        mask = mask.squeeze(-1)

        mask = torch.flip(mask, dims=(0,))
        mask = mask.cpu().clamp(0, 1).numpy()

        mask_u8 = (mask * 255).astype(np.uint8)
        mask_list.append(mask_u8)

        torch.cuda.empty_cache()

    tmp_dir = os.path.join(output_dirname, "tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    if dump_patches:
        os.makedirs(tmp_dir)
        for idx, mask in enumerate(mask_list):
            # === Save final atlas images
            iio.imwrite(os.path.join(tmp_dir, f"mask_{idx}.png"), mask)

    mask_all = np.array(mask_list).reshape(len(mask_list), -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(mask_all)

    # === Find centroid images
    centroid_indices = []
    for k in range(n_clusters):
        cluster_mask = (labels == k)
        cluster_points = mask_all[cluster_mask]

        if len(cluster_points) == 0:
            continue

        # Compute distance from cluster center
        center = kmeans.cluster_centers_[k].copy()
        center[center > 0] = 1.0
        distances = np.sum(np.minimum(cluster_points, center), axis=1)
        # distances = np.linalg.norm(cluster_points - center, axis=1)

        # Index within cluster → index within mask_all
        cluster_indices = np.where(cluster_mask)[0]
        centroid_idx = cluster_indices[np.argmin(distances)]
        centroid_indices.append(centroid_idx)

    # === Save centroid images
    centroid_dir = os.path.join(output_dirname, "centroids")
    os.makedirs(centroid_dir, exist_ok=True)

    for i, idx in enumerate(centroid_indices):
        mask = mask_list[idx]
        out_path = os.path.join(centroid_dir, f"centroid_{i}.png")
        iio.imwrite(out_path, mask)

    print(f"[✅] Saved {len(centroid_indices)} centroid images to {centroid_dir}")

    return centroid_indices, labels











# Note: the part below was originally in a separate file.

from PIL import Image, ImageDraw, ImageFont

def tesselate_wisp(wisp, uv_rect=None, margin_top=1e-4, use_arclength_uv=False):
    """
    Convert one wisp into vertices, triangles, and UVs within the given UV box.
    Allows margin control per edge. For now, only top (v1) margin is used.
    UVs are center-sampled within the box to align with texel centers.

    Args:
        wisp (np.ndarray): Shape (S, N, 3) representing strands in a wisp.
        uv_rect (tuple, optional): UV bounding box (u0, v0, u1, v1).
        margin_top (float): Margin at the top edge (v1).
        use_arclength_uv (bool): Whether to space UVs based on actual strand lengths.

    Returns:
        tuple: verts (np.ndarray), faces (np.ndarray), uvs (np.ndarray, optional)
    """

    S, N, _ = wisp.shape
    verts = wisp.reshape(-1, 3)

    # Generate faces
    faces = []
    for i in range(S - 1):
        for j in range(N - 1):
            i0 = i * N + j
            i1 = (i + 1) * N + j
            i2 = (i + 1) * N + j + 1
            i3 = i * N + j + 1
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])
    faces = np.array(faces, dtype=np.int32)

    if uv_rect is None:
        return verts, faces

    u0, v0, u1, v1 = uv_rect
    v1 -= margin_top

    uu = np.linspace(u0, u1, S)

    if use_arclength_uv:
        # Compute arclength-based spacing along strands
        strand_lengths = np.linalg.norm(np.diff(wisp, axis=1), axis=2).sum(axis=1)
        cumulative_lengths = np.zeros((S, N))
        cumulative_lengths[:, 1:] = np.cumsum(np.linalg.norm(np.diff(wisp, axis=1), axis=2), axis=1)
        normalized_lengths = cumulative_lengths / cumulative_lengths[:, -1][:, None]
        vv = v0 + (v1 - v0) * normalized_lengths
    else:
        vv = np.linspace(v0, v1, N)
        vv = np.tile(vv, (S, 1))

    uu, _ = np.meshgrid(uu, np.arange(N), indexing='ij')
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)

    return verts, faces, uvs

def combine_wisps(wisps, layout=None, use_arclength_uv=False):
    """
    Tesselate all wisps and merge into a single mesh.
    If layout is None, UVs are skipped and only vertices and faces are returned.

    Args:
        wisps (list): List of wisps.
        layout (list, optional): UV bounding boxes.
        use_arclength_uv (bool): Use arclength for UV mapping.

    Returns:
        tuple: vertices, faces, (optional) uvs
    """
    all_vertices, all_faces, all_uvs = [], [], []
    offset = 0

    if layout is None:
        for wisp in wisps:
            verts, faces = tesselate_wisp(wisp, uv_rect=None, use_arclength_uv=use_arclength_uv)
            all_vertices.append(verts)
            all_faces.append(faces + offset)
            offset += verts.shape[0]

        return (
            np.concatenate(all_vertices, axis=0),
            np.concatenate(all_faces, axis=0),
        )

    for idx, (_, u0, v0, u1, v1) in enumerate(layout):
        verts, faces, uvs = tesselate_wisp(
            wisps[idx], (u0, v0, u1, v1), use_arclength_uv=use_arclength_uv
        )
        all_vertices.append(verts)
        all_faces.append(faces + offset)
        all_uvs.append(uvs)
        offset += verts.shape[0]

    return (
        np.concatenate(all_vertices, axis=0),
        np.concatenate(all_faces, axis=0),
        np.concatenate(all_uvs, axis=0),
    )

def wisp_to_grid_mesh(wisp, uv_rect=None, margin_top=1e-4):
    """
    Convert one wisp (set of nearly parallel polylines) into vertices, triangles, UVs, 
    and additional quad-based topological information.

    Args:
        wisp (np.ndarray): (S, N, 3) array representing S polylines with N points each.
        uv_rect (tuple): (u0, v0, u1, v1) UV bounding box for generating UV coordinates.
        margin_top (float): Top margin applied to the UV box.

    Returns:
        verts (np.ndarray): (S*N, 3) vertex array.
        faces (np.ndarray): (num_faces, 3) triangle faces.
        uvs (np.ndarray): (S*N, 2) UV coordinates.
        quads (np.ndarray): (num_quads, 4) indices of vertices per quad (CCW order).
        h_edges (np.ndarray): (num_h_edges, 2) horizontal edge indices (along original polylines).
        w_edges (np.ndarray): (num_w_edges, 2) width edge indices (connecting across polylines).
    """
    S, N, _ = wisp.shape
    verts = wisp.reshape(-1, 3)

    # Generate triangle faces
    faces = []
    for i in range(S - 1):
        for j in range(N - 1):
            i0 = i * N + j
            i1 = (i + 1) * N + j
            i2 = (i + 1) * N + j + 1
            i3 = i * N + j + 1
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])
    faces = np.array(faces, dtype=np.int32)

    # Generate quads (CCW order)
    quads = []
    for i in range(S - 1):
        for j in range(N - 1):
            i0 = i * N + j
            i1 = (i + 1) * N + j
            i2 = (i + 1) * N + j + 1
            i3 = i * N + j + 1
            quads.append([i0, i1, i2, i3])
    quads = np.array(quads, dtype=np.int32)

    # Horizontal edges (along original polylines)
    h_edges = []
    for i in range(S):
        for j in range(N - 1):
            idx0 = i * N + j
            idx1 = i * N + (j + 1)
            h_edges.append([idx0, idx1])
    h_edges = np.array(h_edges, dtype=np.int32)

    # Width edges (across polylines)
    w_edges = []
    for i in range(S - 1):
        for j in range(N):
            idx0 = i * N + j
            idx1 = (i + 1) * N + j
            w_edges.append([idx0, idx1])
    w_edges = np.array(w_edges, dtype=np.int32)

    if uv_rect is None:
        return verts, faces, quads, h_edges, w_edges

    # UV generation logic (copied exactly)
    u0, v0, u1, v1 = uv_rect
    v1 -= margin_top

    uu = np.linspace(u0, u1, S)
    vv = np.linspace(v0, v1, N)
    uu, vv = np.meshgrid(uu, vv, indexing='ij')
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)

    return verts, faces, uvs, quads, h_edges, w_edges

def combine_wisps_with_cluster_labels(wisps, layout, wisp2cluster):
    """
    Tesselate and combine multiple wisps into a single mesh, assigning each face
    a cluster label based on the corresponding wisp.

    Args:
        wisps (List[np.ndarray]):
            List of wisps, each defined as a (num_points, 3) array of 3D points.
        layout (List[Tuple]):
            List containing tuples of the form (idx, u0, v0, u1, v1), specifying the
            UV coordinates for each wisp's texture region.
        wisp2cluster (List[int]):
            List mapping each wisp index to its corresponding cluster ID.

    Returns:
        vertices (np.ndarray):
            Concatenated vertices of all wisps, shape (total_vertices, 3).
        faces (np.ndarray):
            Concatenated triangle indices, shape (total_faces, 3).
        uvs (np.ndarray):
            Concatenated UV coordinates, shape (total_vertices, 2).
        face2cluster (np.ndarray):
            Array mapping each face to its cluster ID, shape (total_faces,).
        quads (np.ndarray):
            Concatenated quad indices, shape (total_quads, 4).
        h_edges (np.ndarray):
            Concatenated horizontal edge indices, shape (total_h_edges, 2).
        w_edges (np.ndarray):
            Concatenated width edge indices, shape (total_w_edges, 2).
    """
    all_vertices, all_faces, all_uvs = [], [], []
    all_quads, all_h_edges, all_w_edges = [], [], []
    face2cluster = []
    offset = 0

    for idx, (_, u0, v0, u1, v1) in enumerate(layout):
        verts, faces, uvs, quads, h_edges, w_edges = wisp_to_grid_mesh(wisps[idx], (u0, v0, u1, v1))
        all_vertices.append(verts)
        all_faces.append(faces + offset)
        all_uvs.append(uvs)
        all_quads.append(quads + offset)
        all_h_edges.append(h_edges + offset)
        all_w_edges.append(w_edges + offset)
        face2cluster.append(np.full((faces.shape[0],), wisp2cluster[idx]))
        offset += verts.shape[0]

    return (
        np.concatenate(all_vertices, axis=0),
        np.concatenate(all_faces, axis=0),
        np.concatenate(all_uvs, axis=0),
        np.concatenate(face2cluster, axis=0),
        np.concatenate(all_quads, axis=0),
        np.concatenate(all_h_edges, axis=0),
        np.concatenate(all_w_edges, axis=0),
    )

def generate_uv_texture(tex_filename, layout, resolution=4096):
    """
    Generate a UV image with random solid colors per wisp and render its index number.
    """
    os.makedirs(os.path.dirname(tex_filename), exist_ok=True)

    # Create texture
    tex = Image.new("RGB", (resolution, resolution), color=(255, 255, 255))
    draw = ImageDraw.Draw(tex)

    try:
        font = ImageFont.truetype("arial.ttf", size=int(resolution * 0.02))
    except IOError:
        font = ImageFont.load_default()

    np.random.seed(123)
    for idx, (sorted_idx, u0, v0, u1, v1) in enumerate(layout):
        color = tuple((np.random.rand(3) * 255).astype(np.uint8))
        x0 = int(u0 * resolution)
        y0 = int((1 - v1) * resolution)
        x1 = int(u1 * resolution)
        y1 = int((1 - v0) * resolution)

        # Fill region
        draw.rectangle([x0, y0, x1, y1], fill=color)

        # Draw index number
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        draw.text((cx, cy), str(sorted_idx), fill=(0, 0, 0), anchor="mm", font=font)

    tex = tex.transpose(Image.FLIP_TOP_BOTTOM)
    tex.save(tex_filename)
    print(f"🖼️ Saved texture with wisp indices to {tex_filename}")
    return np.asarray(tex).astype(np.float32) / 255.0
