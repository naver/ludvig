import torch
import os
import numpy as np
from utils.graph import energy_fn, normalize_torch, query_neighbors
from utils.image import save_img
from utils.visualization import viz_normalization


class GraphDiffusion:

    def __init__(
        self,
        gaussian,
        render_fn,
        cameras,
        logdir,
        num_neighbors,
        num_iterations,
        feature_bandwidth=None,
        trace_name=None,
        eps=1e-8,
    ):
        """
        Base class for running graph diffusion based on 3D features associated with a Gaussian Splatting scene.

        Args:
            gaussian (object): An instance of `gaussiansplatting.scene.gaussian_model.GaussianModel`.
            render_fn (callable): A function for rendering 3D features, used for visualizations.
            cameras (iterable): A list of cameras instance, e.g. `gaussiansplatting.scene.cameras.Simple_Camera`.
            logdir (str or None): Path to the directory where diffusion logs are saved.
            num_neighbors (int): The number of neighbors to consider in the k-NN graph.
            num_iterations (int): The number of diffusion iterations to perform.
            feature_bandwidth (float, optional): Bandwidth parameter for computing feature similarities.
            trace_name (str, optional): Camera name for visualizations.
            eps (float, optional): A small value added for numerical stability during normalization.
        """
        self.mask = None
        self.gaussian = gaussian
        self.cameras = cameras
        self.render_fn = render_fn
        self.knn_neighbor_indices = None
        self.num_neighbors = num_neighbors
        self.num_iterations = num_iterations
        self.feature_bandwidth = feature_bandwidth
        self.trace_name = trace_name
        self.logdir = logdir
        self.eps = eps

    def compute_knn_graph(self):
        """Extract K nearest euclidean neighbors for each Gaussian."""
        self.knn_neighbor_indices, _ = query_neighbors(
            self.gaussian._xyz.detach(),
            num_neighbors=self.num_neighbors,
        )

    def __call__(self, features, **kwargs):
        if self.knn_neighbor_indices is None:
            self.compute_knn_graph()
            features = self.normalize_features(features)
            self.precompute_similarities(features)
        features = self.compute_similarities()
        features = self.run_diffusion(features, **kwargs)
        return features

    def normalize_features(self, features):
        feature_norms = self.eps + features.norm(dim=-1, keepdim=True)
        return features / feature_norms

    def precompute_similarities(self, features):
        """Computes pairwise distances between the features of each node and its K neighbors."""
        self.similarities = torch.norm(
            features[:, None] - features[self.knn_neighbor_indices], dim=-1
        )

    def compute_similarities(self):
        """RBF function over precomputed pairwise distances."""
        feature_similarities = energy_fn(
            self.similarities, self.feature_bandwidth, self.mask
        )
        return feature_similarities

    def run_diffusion(self, similarities, unary_term=None, **kwargs):
        """
        Performs the diffusion process based on the computed similarities.

        Args:
            similarities (torch.Tensor): Similarities between each node and its K neighbors.
            unary_term (torch.Tensor, optional): Unary regularization per node and per feature dimension.

        Returns:
            torch.Tensor: The features after diffusion processing.
        """
        if unary_term is None:
            Dleft = torch.sum(similarities, dim=1, keepdim=True) + self.eps
            Dright = torch.sum(similarities, dim=0, keepdim=True) + self.eps
            similarities /= torch.sqrt(Dleft) * torch.sqrt(Dright)
        node_indices = np.arange(self.knn_neighbor_indices.shape[0]).repeat(
            self.knn_neighbor_indices.shape[1], axis=0
        )
        combined_indices = np.stack((node_indices, self.knn_neighbor_indices.flatten()))
        processed_features = self.get_stationary(
            combined_indices,
            similarities.flatten(),
            unary_term=unary_term,
            normalize=unary_term is not None,
            **kwargs
        )
        return processed_features

    def get_stationary(
        self,
        neighbors,
        similarities,
        normalize=False,
        normalize_f=True,
        f=None,
        binarize=None,
        unary_term=None,
        symmetrize=True
    ):
        """
        Constructs adjacency matrix based on feature similarities and neighbor indices and runs graph diffusion.

        Args:
            neighbors (numpy.ndarray): Indices of node-to-node connections in the graph.
            similarities (torch.Tensor): Edge weights corresponding to the `neighbors` indices.
            normalize (bool, optional): Whether to normalize the graph adjacency matrix.
            normalize_f (bool, optional): Whether to normalize features at each iteration (power method).
            f (torch.Tensor, optional): Initial feature values for the nodes.
            binarize (float, optional): A threshold to binarize similarities.
                If set, similarities greater than this value are converted to 1, and others to 0.
            unary_term (torch.Tensor, optional): Regularization per node and per dimension.
            symmetrize (bool, optional): Whether to symmetrize the adjacency matrix.

        Returns:
            torch.Tensor: The features after graph diffusion.
        """
        n = self.knn_neighbor_indices.shape[0]
        if f is None:
            f = self.initial_features

        if binarize:
            similarities = (similarities>binarize).type(torch.float32)

        row_indices = torch.tensor(neighbors[0])
        col_indices = torch.tensor(neighbors[1])

        if symmetrize:
            row_sym = torch.cat([row_indices, col_indices])
            col_sym = torch.cat([col_indices, row_indices])
            values_sym = torch.cat([similarities, similarities])
            A_coo = torch.sparse_coo_tensor(
                torch.stack([row_sym, col_sym]),
                values_sym,
                (n, n),
                device="cuda",
                dtype=f.dtype
            ).coalesce()
            A = A_coo.to_sparse_csr()
        else:
            crow_indices = torch._convert_indices_from_coo_to_csr(row_indices, n)
            A = torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                similarities,
                (n, n),
                dtype=f.dtype,
                device="cuda"
            )
        if normalize:
            if unary_term is not None:
                unary_term /= torch.sqrt(unary_term) * (A @ torch.sqrt(unary_term)) + 1e-8
            else:
                A = normalize_torch(A, n)
        for i in range(self.num_iterations):
            if normalize_f:
                f /= self.eps + f.norm(dim=0, keepdim=True)
            if unary_term is not None:
                f = torch.sqrt(unary_term) * (A @ (torch.sqrt(unary_term) * f))
            else:
                f = A @ f
            # self.trace_f(f, i)
        return f

    def trace_to_camera(self):
        cam = None
        if isinstance(self.trace_name, int):
            cam = self.cameras[self.trace_name]
        if isinstance(self.trace_name, str):
            cam = next(
                (cam for cam in self.cameras if cam.image_name == self.trace_name), None
            )
        assert cam is not None, f"Did not find camera for name {self.trace_name}."
        return cam

    def trace_f(self, f, t):
        if self.trace_name is None:
            return
        cam = self.trace_to_camera()
        tstr = str(t).zfill(3)
        feat = self.render_fn(f.sum(1, keepdim=True).repeat(1, 3), cam)[:1]
        diffusion_dir = os.path.join(self.logdir, "diffusion")
        os.makedirs(diffusion_dir, exist_ok=True)
        save_img(
            os.path.join(diffusion_dir, f"t{tstr}.png"),
            viz_normalization(feat).squeeze(),
            text=f"Step {t+1}",
            font_size=60,
        )
