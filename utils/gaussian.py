import torch
import math


def prune(gaussian, minp=0.05, maxp=0.95):
    L = gaussian.get_covariance().detach()
    cov = covariance_from_stripped(L)
    n0 = len(gaussian._xyz)
    V = torch.abs(torch.det(cov))
    keep = torch.where(
        (V < V.quantile(maxp, dim=-1, keepdim=True))
        * (V > V.quantile(minp, dim=-1, keepdim=True))
    )
    gaussian.prune_points_noopt(keep)
    n1 = len(gaussian._xyz)
    print(f"Kept {n1} points out of {n0}.")
    format_fn = lambda x: "{:.2e}".format(x.item())
    print("Min-max volume before pruning:", format_fn(V.min()), format_fn(V.max()))
    print("Min-max volume after pruning:", format_fn(V[keep].min()), format_fn(V[keep].max()))
    return keep


def covariance_from_stripped(L):
    covariance = torch.zeros((len(L), 3, 3), dtype=torch.float32, device="cuda")
    covariance[:, 0, 1] = L[:, 1]
    covariance[:, 0, 2] = L[:, 2]
    covariance[:, 1, 2] = L[:, 4]
    covariance = covariance + covariance.transpose(1, 2)
    covariance[:, 0, 0] = L[:, 0]
    covariance[:, 1, 1] = L[:, 3]
    covariance[:, 2, 2] = L[:, 5]
    return covariance
