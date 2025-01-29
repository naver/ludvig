import torch
from utils.graph import energy_fn
from utils.scribble import scribble_inverse_rendering
from sklearn.linear_model import LogisticRegression
from .base import GraphDiffusion


class GraphDiffusionSeg(GraphDiffusion):

    def __init__(
        self,
        *args,
        scribbles_2d,
        scribble_camera,
        maxpos,
        reg_bandwidth=None,
        logreg=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.initial_features = None
        self.scribbles_2d = scribbles_2d
        self.scribble_camera = scribble_camera
        self.maxpos = maxpos
        self.reg_bandwidth = reg_bandwidth
        self.logreg = logreg
        self.logreg_fn = None
        self.compute_knn_graph()

    def __call__(self, features):
        features = self.normalize_features(features)
        if self.initial_features is None:
            self.compute_initial_features()
            self.mask = self.initial_features.squeeze() > 0
            self.precompute_similarities(features)
        similarities = self.compute_similarities()
        self.compute_regularizer(features)
        similarities *= torch.sqrt(
            self.reg_similarities[self.knn_neighbor_indices] * self.reg_similarities[:, None]
        )
        diffused_features = self.run_diffusion(similarities, binarize=1e-5)
        diffused_features = (diffused_features>0) * self.reg_similarities[:,None].type(torch.float32)
        return diffused_features, self.reg_similarities

    def compute_initial_features(self):
        mask, self.Z = scribble_inverse_rendering(
            self.scribbles_2d, self.gaussian, self.scribble_camera, return_counts=True
        )
        npos = (mask > 0).sum()
        if self.maxpos > 0:
            a = torch.argsort(mask.squeeze())
            mask[a[: -int(self.maxpos * npos)]] = 0
        self.initial_features = mask

    def compute_regularizer(self, features):
        if self.logreg:
            sample_weight = self.Z
            fn = features.cpu().numpy()
            if self.logreg_fn is None:
                reference_features = self.initial_features.squeeze()
                if torch.any(reference_features < 0):
                    pmask = (reference_features > 0) + (reference_features.squeeze() < 0)
                    fn_ = features[pmask].cpu().numpy()
                    sample_weight_ = sample_weight[pmask]
                    labels_ = (reference_features > 0)[pmask]
                    print("Proportion of positives:", pmask.sum() / len(pmask))
                else:
                    pmask = reference_features > 0
                    print("Prop positives:", pmask.sum() / len(pmask))
                    fn_ = fn
                    labels_ = pmask
                    sample_weight_ = sample_weight
                # fn2 = (fn[:, :, None] @ fn[:, None]).view(len(fn), -1)
                # fn = torch.cat((fn, fn2), dim=1).cpu().numpy()
                self.logreg_fn = LogisticRegression(
                    C=self.logreg, class_weight="balanced"
                ).fit(
                    fn_,
                    labels_.cpu().numpy().astype(int),
                    sample_weight=sample_weight_.squeeze().cpu().numpy(),
                )
            reg_similarities = torch.from_numpy(
                self.logreg_fn.predict_proba(fn)
            ).cuda()[:, 1]
            reg_similarities = reg_similarities ** (1 / self.reg_bandwidth)
        else:
            reg = features[self.initial_features.squeeze() > 0].mean(0)
            reg_similarities = torch.norm(features - reg[None], dim=-1)
            reg_similarities = energy_fn(
                reg_similarities, self.reg_bandwidth, self.mask
            )
        self.reg_similarities = reg_similarities
