import math
import torch
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def energy_fn(x, bandwidth, mask=None, dim=None):
    if mask is None:
        median = x.median() if dim is None else x.median(dim=dim, keepdim=True).values
    else:
        median = x[mask].median() if dim is None else x[mask].median(dim=dim, keepdim=True).values
    return torch.exp(-(x**2) / (bandwidth * median**2))


def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    vals, vecs = torch.eig(matrix, eigenvectors=True)
    vals_pow = vals.pow(p)
    matrix_pow = torch.matmul(
        vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs))
    )
    return matrix_pow


def positional_encoding_3d(positions, embedding_dim):
    assert embedding_dim % 6 == 0, "Embedding dimension must be divisible by 6"
    pe = torch.zeros(positions.shape[0], embedding_dim, device="cuda")
    div_term = torch.exp(
        torch.arange(0, embedding_dim // 6, dtype=torch.float32, device="cuda")
        * (-math.log(10000.0) / (embedding_dim // 6))
    )

    # For each dimension x, y, z, compute the positional encodings
    for i in range(positions.shape[1]):
        pe[:, i * 2 * (embedding_dim // 6) : (i + 1) * 2 * (embedding_dim // 6) : 2] = (
            torch.sin(positions[:, i].unsqueeze(1) * div_term)
        )
        pe[
            :, i * 2 * (embedding_dim // 6) + 1 : (i + 1) * 2 * (embedding_dim // 6) : 2
        ] = torch.cos(positions[:, i].unsqueeze(1) * div_term)

    return pe


def create_adjacency_matrix(similarities, knn_indices):
    n = len(similarities)
    similarities = similarities[np.arange(n)[:, None], knn_indices]
    batch_indices = torch.arange(n, device="cuda").view(-1, 1).expand_as(knn_indices)
    adjacency_matrix = torch.zeros((n, n), dtype=torch.float32, device="cuda")
    adjacency_matrix[batch_indices, knn_indices] = similarities
    adjacency_matrix[knn_indices, batch_indices] = (
        similarities  # Ensure the graph is undirected
    )
    return adjacency_matrix


def normalize_torch(A, n):
    degree = A @ torch.ones(A.shape[1], device="cuda")
    degree_inv_sqrt = 1.0 / torch.sqrt(degree)
    print("degree_inv_sqrt", degree_inv_sqrt.dtype)
    D_inv_sqrt = csr_diags(degree_inv_sqrt)
    A = D_inv_sqrt @ A @ D_inv_sqrt
    return A


def get_stationary(
    f, neighbors, distances, n, p=6, normalize=True, mixing=1.0, normalize_f=True
):
    crow_indices = torch._convert_indices_from_coo_to_csr(
        torch.tensor(neighbors[0]), f.shape[0]
    )
    A = torch.sparse_csr_tensor(
        crow_indices,
        neighbors[1],
        distances,
        (n, n),
        dtype=f.dtype,
        device="cuda",
    )
    if normalize:
        A = normalize_torch(A, n)
    # import pdb; pdb.set_trace()
    for i in range(p):
        if normalize_f:
            f /= 1e-8 + f.norm(dim=-1, keepdim=True)
        f = mixing * A @ f + (1 - mixing) * f
        # f = (f - f.mean()) / f.std()
    # import pdb; pdb.set_trace()
    return f


def csr_diags(values):
    n = len(values)
    crow_indices = torch.arange(n + 1, dtype=torch.int64)
    col_indices = torch.arange(n, dtype=torch.int64)
    matrix = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, size=(n, n), device="cuda"
    )
    return matrix


def query_neighbors(points, num_neighbors=10000, k=None, include_self=True):
    """
    Finds the nearest neighbors for each point in a point cloud.

    Parameters:
    points (torch.Tensor): A tensor of shape (N, D) representing N points in D dimensions.
    num_neighbors (int): The number of nearest neighbors to query.
    k (int or None): The number of neighbors to keep randomly. If None, keep all `q` neighbors.

    Returns:
    ind (np.ndarray): An array of shape (N, k) or (N, q) containing the indices of the nearest neighbors.
    dpos (np.ndarray): An array of shape (N, k) or (N, q) containing the distances to the nearest neighbors.
    """
    print("Generating edges from point cloud...")

    # Convert the input tensor to a numpy array
    points_np = points.cpu().numpy()

    # Build a k-d tree using the points
    kdtree = cKDTree(points_np)

    neighbor_indices = []
    neighbor_distances = []

    msg =  f"Querying {num_neighbors} euclidean neighbors for each of the {len(points_np)} Gaussians."
    for i, point in tqdm(
        enumerate(points_np),
        total=len(points_np),
        bar_format=f'{{n_fmt}}/{{total_fmt}} | Elapsed: {{elapsed}}s | {msg}',
        mininterval=0.5
    ):
        # Find the q+1 nearest neighbors (including the point itself)
        distances, indices = kdtree.query(point, k=num_neighbors + 1)

        if k is not None:
            # Randomly select k neighbors (excluding the point itself at index 0)
            selected_indices = np.random.choice(
                np.arange(1-include_self, num_neighbors + 1), k, replace=False
            )
            neighbor_indices.append(indices[selected_indices])
            neighbor_distances.append(distances[selected_indices])
        else:
            # Keep all q neighbors (excluding the point itself at index 0)
            if include_self is False:
                neighbor_indices.append(indices[1:])
                neighbor_distances.append(distances[1:])
            else:
                neighbor_indices.append(indices)
                neighbor_distances.append(distances)

    # Convert the lists to numpy arrays
    neighbor_indices = np.array(neighbor_indices).astype(int)
    neighbor_distances = np.array(neighbor_distances)

    return neighbor_indices, neighbor_distances
