import torch
import time
from tqdm import tqdm
from .image import resize


@torch.no_grad()
def uplifting(
    loader, gaussian, resolution=None, prune_gaussians=None, min_gaussians=400000
):
    """
    Performs 2D-to-3D uplifting by weighted aggregation of pixel features.

    This function processes 2D image features (from a data loader) and aggregates
    them into 3D Gaussian features based on camera projections. Optionally, it resizes
    the input features and prunes the Gaussians based on their importance.

    Args:
        loader (iterable or list): Contains pairs of (features, camera), with
            - features a 2D feature map of shape (C, H, W)
            - camera an instance of gaussiansplatting.scene.cameras.Simple_Camera
        gaussian (object): an instance of gaussiansplatting.scene.gaussian_model.GaussianModel
        resolution (tuple of int, optional): Target resolution (height, width) to resize
            the input features if their resolution differs from this. Defaults to None.
        prune_gaussians (int or float, optional): Pruning parameter to reduce the number
            of Gaussians:
            - If `int`, specifies the exact number of Gaussians to retain based on weights.
            - If `float` (between 0 and 1), specifies the quantile threshold to prune Gaussians
              with low weights. Defaults to None (no pruning).
        min_gaussians (int, optional): Minimum number of Gaussians to keep.

    Returns:
        tuple:
            - features_3d (torch.Tensor): A tensor containing the aggregated 3D features
              for each Gaussian.
            - keep (torch.Tensor or None): Indices of the retained Gaussians if pruning is applied,
              otherwise None.
    """
    weights = torch.zeros_like(gaussian._opacity, dtype=torch.float32)
    t0 = time.time()
    features_3d = None
    if isinstance(loader, list):
        loader = iter(loader)
    for j, (feat, cam) in tqdm(enumerate(loader)):
        if len(feat.shape) == 4:
            feat = feat.squeeze(0)
        if j == 0:
            features_3d = torch.zeros(
                (len(gaussian._opacity), len(feat)),
                dtype=torch.float32,
                device="cuda",
            )
        if resolution and feat.shape[-1] != resolution[-1]:
            feat = resize(feat, resolution)
        gaussian.apply_weights(cam, features_3d, weights, feat.cuda())
    features_3d /= weights + 1e-8
    keep = None
    total_time = time.time() - t0
    print(f"Time for uplifting {j+1} views: {round(total_time, 1)}s")
    if prune_gaussians is not None and len(gaussian._xyz) > min_gaussians:
        if isinstance(prune_gaussians, int):
            sorting = torch.argsort(weights.squeeze())
            keep = sorting[-prune_gaussians:]
        else:
            keep = torch.where(
                weights.squeeze() > weights.squeeze().quantile(prune_gaussians)
            )[0]
        print(f"Keeping {len(keep)} points out of {len(weights)}.")
        gaussian.prune_points_noopt(keep)
        features_3d = features_3d[keep]
    return features_3d, keep
