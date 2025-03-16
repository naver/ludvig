import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from .base import GraphDiffusion
from utils.image import save_img
from utils.visualization import viz_normalization
from utils.evaluation import multi_thresholding
from utils.graph import energy_fn as energy_fn_base
from skimage import filters


class GraphDiffusionCLIP(GraphDiffusion):

    def __init__(
        self, *args, load_dino, relev, reg_bandwidth=None, logreg=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.load_dino = load_dino
        self.relev = relev
        self.reg_bandwidth = reg_bandwidth
        self.logreg_fn = None
        self.logreg = logreg
        self.Z = self.compute_Z()
        self.construct()

    def __call__(self, **kwargs):
        if self.knn_neighbor_indices is None:
            self.compute_knn_graph()
            self.precompute_weights()
        similarities = self.compute_similarities()
        reg_similarities = self.compute_regularizer()
        diffused_features = self.run_diffusion(similarities, unary_term=reg_similarities, **kwargs)
        return diffused_features

    def construct(self):
        a = self.relev.min(dim=1, keepdim=True).values
        b = self.relev.max(dim=1, keepdim=True).values
        self.nrel = (self.relev - a) / (b - a)
        self.mask = torch.stack(
            [r > filters.threshold_otsu(r[r > .5].cpu().numpy()) for r in self.nrel]
        )
        self.initial_features = torch.stack(
            [
                r * (r >= min(r.quantile(.999).item(), multi_thresholding(r[m].cpu().numpy(), degree=2)))
                for r, m in zip(self.relev, self.mask)
            ]
        ).T
        print(
            "Number of positive nodes per prompt at graph initialization:",
            list(map(int, (self.initial_features>0).sum(dim=0).cpu().numpy()))
        )

    def compute_similarities(self):
        return self.energy_fn(self.similarities, self.feature_bandwidth)

    def precompute_weights(self):
        dinov2_features = torch.from_numpy(np.load(self.load_dino)).cuda()
        dinov2_features = self.normalize_features(dinov2_features)
        self.precompute_similarities(dinov2_features)
        if self.logreg:
            dinov2_features = dinov2_features.cpu().numpy()
            probas = []
            for mask in self.mask:
                logreg = LogisticRegression(
                    C=self.logreg, max_iter=1000, class_weight="balanced"
                ).fit(
                    dinov2_features,
                    mask.cpu().numpy().astype(int),
                    sample_weight=self.Z.squeeze().cpu().numpy(),
                )
                probas.append(logreg.predict_proba(dinov2_features)[:, 1])
            self.probas = torch.from_numpy(np.stack(probas)).cuda().type(torch.float32)
        else:
            masked_relev = (self.relev * self.mask)[:, :, None]
            average_features = (dinov2_features[None] * masked_relev).sum(
                dim=1
            ) / masked_relev.sum(dim=1)
            average_features /= 1e-6 + average_features.norm(dim=-1, keepdim=True)
            self.probas = torch.einsum(
                "md,nd->mn", average_features, dinov2_features
            )
            self.probas = (2-2*self.probas)**.5

    def compute_Z(self):
        feat_3d = torch.ones_like(self.gaussian._xyz)
        weights = torch.zeros_like(self.gaussian._opacity)
        feat_2d = torch.ones_like(self.render_fn(feat_3d, self.cameras[0]))
        for cam in self.cameras:
            self.gaussian.apply_weights(cam, feat_3d, weights, feat_2d)
        return weights / len(self.cameras)

    def energy_fn(self, x, bandwidth):
        median = torch.zeros(len(x), device="cuda", dtype=torch.float32)[:, None]
        amax = torch.argmax(self.relev, dim=0)
        for n in range(len(self.relev)):
            if torch.any(self.mask[n]):
                median[amax == n] = x[self.mask[n]].median()
        return torch.exp(-(x**2) / (bandwidth * median**2 + 1e-8))

    def compute_regularizer(self):
        if self.logreg:
            return (self.probas ** (1 / self.reg_bandwidth)).T
        return energy_fn_base(self.probas, self.reg_bandwidth, dim=1).T
