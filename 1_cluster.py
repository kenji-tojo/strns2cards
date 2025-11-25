import numpy as np
import torch
import pystrns2cards as s2c
import os
import argparse
import polyscope as ps

np.random.seed(42)

NUM_SAMPLES_PER_STRAND = 100

def run_one_model(args, model_name):
    strands_path = os.path.join("./dataset/Groom", args.dataset, "npz", f"{model_name}.npz")

    strands = s2c.load_npz_as_strands_arc_length(strands_path, target_num_samples=NUM_SAMPLES_PER_STRAND)
    scale = np.array([1, -1, 1], dtype=np.float32)
    strands = [v * scale for v in strands]
    original_num_strands = len(strands)

    print(f">>> Processing: {model_name} ({len(strands)} strands)")

    all_points = np.concatenate(strands, axis=0)
    min_xyz = all_points.min(axis=0)
    max_xyz = all_points.max(axis=0)
    bbox_size = max_xyz - min_xyz

    strands_np = np.stack(strands, axis=0)
    features = s2c.compute_strand_features(strands_np, n_components=16)

    _, label = s2c.cluster_points_kmeans(points=features, num_clusters=args.num_clusters)

    # === Cluster filtering: remove outliers and build wisps ===
    indices = np.arange(len(strands), dtype=np.int64)
    wisps = []
    cluster_ids = []
    source_indices = []
    for cluster_id in range(args.num_clusters):
        cluster_mask = (label == cluster_id)                # (N,)
        cluster_strands = strands_np[cluster_mask]          # (Nc, S, 3)
        cluster_indices = indices[cluster_mask]             # (Nc,)
        root_points = cluster_strands[:, 0, :]              # (Nc, 3)
        keep_mask = s2c.statistical_outlier_mask(root_points, n_neighbors=5, ratio=2.0)
        filtered = cluster_strands[keep_mask]               # (Nc', S, 3)
        filtered_indices = cluster_indices[keep_mask]       # (Nc',)
        wisps.append(filtered)
        source_indices.append(filtered_indices)
        cluster_ids.append(np.full(len(filtered), cluster_id, dtype=np.int64))
    strands_np = np.concatenate(wisps, axis=0)
    source_indices = np.concatenate(source_indices, axis=0)
    cluster_ids = np.concatenate(cluster_ids, axis=0)

    del strands, all_points, label, indices

    diffs = strands_np[:, 1:, :] - strands_np[:, :-1, :]
    lengths = np.linalg.norm(diffs, axis=2).sum(axis=1)
    sorted_indices = np.argsort(-lengths)
    seen = set()
    guide_indices = []
    for i in sorted_indices:
        cluster_id = cluster_ids[i]
        if cluster_id not in seen:
            guide_indices.append(i)
            seen.add(cluster_id)
            if len(seen) == args.num_clusters:
                break
    guide_indices = np.array(guide_indices)
    # Ensure guide_indices are sorted by ascending cluster ID
    guide_indices = guide_indices[np.argsort(cluster_ids[guide_indices])]

    print(f"Found {len(guide_indices)} clusters and corresponding guide strands")
    # print(f"guide_cluster_ids: {cluster_ids[guide_indices].tolist()}")

    if args.show:
        ps.init()
        all_points = strands_np.reshape(-1, 3)
        rgb_colors = np.random.rand(args.num_clusters, 3)
        vertex_colors = np.concatenate([
            np.tile(rgb_colors[cluster_ids[i]], (len(strand), 1))
            for i, strand in enumerate(strands_np)
        ], axis=0)

        edges = s2c.build_edge_indices_from_wisp(strands_np)
        ps.register_curve_network("hair strands", all_points, edges, radius=2e-3) \
          .add_color_quantity("cluster color", vertex_colors, defined_on='nodes', enabled=True)

        guide_strands_np = strands_np[guide_indices]
        guide_edges = s2c.build_edge_indices_from_wisp(guide_strands_np)
        guide_vertex_colors = np.concatenate([
            np.tile(rgb_colors[cluster_ids[i]], (len(strands_np[i]), 1))
            for i in guide_indices
        ], axis=0)

        ps.register_curve_network("guide strands", guide_strands_np.reshape(-1, 3), guide_edges, radius=5e-3) \
          .add_color_quantity("cluster color", guide_vertex_colors, defined_on='nodes', enabled=True)
        ps.show()
        return

    # === Save outputs ===
    experiment_ID = args.experiment_ID if args.experiment_ID is not None else "example"
    output_root = f"./output/{experiment_ID}"
    config_name = f"cluster{args.num_clusters}"
    output_dirname = os.path.join(output_root, args.dataset, config_name, model_name)
    os.makedirs(output_dirname, exist_ok=True)

    np.savez_compressed(os.path.join(output_dirname, f"{model_name}_data.npz"),
                        source_indices=source_indices,
                        cluster_ids=cluster_ids,
                        guide_indices=guide_indices)

    log_lines = []
    log_lines.append("=== Parameters ===")
    for key, val in vars(args).items():
        log_lines.append(f"{key}: {val}")

    log_lines.append("\n=== Dataset Info ===")
    log_lines.append(f"num_strands (original): {original_num_strands}")
    log_lines.append(f"num_strands (filtered): {len(source_indices)}")
    log_lines.append(f"bbox_min: {min_xyz.tolist()}")
    log_lines.append(f"bbox_max: {max_xyz.tolist()}")
    log_lines.append(f"bbox_size: {bbox_size.tolist()}")
    log_lines.append(f"num_guide_strands: {len(guide_indices)}")

    with open(os.path.join(output_dirname, f"cluster_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Cluster hair strands into wisps")
    parser.add_argument("--experiment_ID", type=str, default="example", help="Experiment name")
    parser.add_argument("--dataset", type=str, default="CT2Hair", help="Dataset name")
    parser.add_argument("--model_name", type=str, default="All", help="Model name, or 'All'")
    parser.add_argument("--num_clusters", type=int, default=300, help="Number of clusters (wisps)")
    parser.add_argument("--show", action="store_true", help="Launch visualizer")
    args = parser.parse_args()

    if args.model_name.lower() == "all":
        npz_dir = os.path.join("./dataset/Groom", args.dataset, "npz")
        model_names = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(npz_dir)
            if f.endswith(".npz") and not f.startswith(".")
        ])

        for model_name in model_names:
            print(f"\n=== Running model: {model_name} ===")
            run_one_model(args, model_name)
    else:
        run_one_model(args, args.model_name)

if __name__ == "__main__":
    main()
