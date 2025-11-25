import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple

def cluster_points_kmeans(
    points: np.ndarray,
    num_clusters: int,
    *,
    tolerance: float = 1e-4,
    init: str = "k-means++",
    yinyang_t: float = 0.1,
    metric: str = "L2",
    seed: int = 42,
    device: int = 0,
    verbosity: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster 3D points into a specified number of clusters using k-means.
    Compatible with libKMCUDA's kmeans_cuda() interface.

    Args:
        points (np.ndarray): (N, D) float32 or float64 array of points.
        num_clusters (int): Desired number of clusters.
        (Additional parameters are accepted but mostly ignored for sklearn fallback.)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - centroids (num_clusters, D) float32
            - cluster_ids (N,) int32
    """
    assert points.ndim == 2, "Input points must have shape (N, D)"

    try:
        from libKMCUDA import kmeans_cuda
        points = points.astype(np.float32, copy=False)
        centroids, cluster_ids = kmeans_cuda(
            points, num_clusters,
            tolerance=tolerance,
            init=init,
            yinyang_t=yinyang_t,
            metric=metric,
            seed=seed,
            device=device,
            verbosity=verbosity
        )
        return centroids, cluster_ids.astype(np.int32)

    except ImportError:
        from sklearn.cluster import KMeans
        print("[s2c] libKMCUDA not found, falling back to sklearn KMeans...")
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto", tol=tolerance, random_state=seed)
        cluster_ids = kmeans.fit_predict(points)
        centroids = kmeans.cluster_centers_
        return centroids.astype(np.float32), cluster_ids.astype(np.int32)


def compute_strand_features(
    strands: np.ndarray,
    n_components: int = 16
) -> np.ndarray:
    """
    Compute feature vectors for a batch of hair strands using PCA.

    Each strand is represented by its root position and a low-dimensional
    PCA embedding of its shape, obtained by subtracting the root and flattening.

    Args:
        strands (np.ndarray): Array of shape (N, S, 3), where N is the number of strands,
                              and S is the number of sample points per strand.
        n_components (int): Number of PCA components to retain.

    Returns:
        np.ndarray: Array of shape (N, 3 + n_components), where the first 3 entries are
                    the root positions and the remaining entries are the PCA features.
    """
    N, S, _ = strands.shape
    strands = strands.copy() - np.mean(strands.reshape(-1, 3), axis=0)
    roots = strands[:, 0, :]
    flat = strands[:, 1:, :].reshape(N, -1)

    features = PCA(n_components=n_components).fit_transform(flat)
    return np.concatenate([roots, features], axis=1)
