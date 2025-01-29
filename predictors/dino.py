import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from dinov2.model import DINOv2
from utils.config import config_to_instance
from utils.image import resize, image_from_path
from .base import BaseDataset

class DINOv2Dataset(BaseDataset):
    def __init__(
        self,
        *args,
        dino_cfg,
        dino_ckpt,
        n_components,
        sliding_window=None,
        eigval_weighting=False,
        use_cuda=True,
        pca_subsample=500000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = DINOv2(dino_cfg, dino_ckpt).cuda()
        self.n_components = n_components
        self.sliding_window = None
        if sliding_window is not None:
            self.sliding_window = config_to_instance(**sliding_window)
        self.eigval_weighting = eigval_weighting
        self.use_cuda = use_cuda
        self.pca_subsample = pca_subsample
        self.extract()

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.features[idx]).cuda()
        if self.sliding_window is not None:
            split_indices = np.cumsum(self.h * self.w * np.ones(self.n_patch).astype(int))
            feat = np.split(feat, split_indices[:-1], axis=0)
            feat = list(map(lambda x: x.T.reshape(self.n_components, self.h, self.w), feat))
            feat = self.sliding_window.fill(
                feat,
                self.sliding_window.indices[idx],
                (self.Hr, self.Wr),
                lambda i, img: resize(img, self.sliding_window.sizes[i]),
            )
        else:
            feat = feat.T.view(self.n_components, self.Hr // 14, self.Wr // 14)
        if self.eigval_weighting:
            feat.mul_(self.eigvals[:,None,None])
        feat = nn.functional.interpolate(
            feat[None], size=(self.H, self.W), mode="bilinear"
        ).squeeze(0)
        return feat, self.cameras[idx]

    def extract(self):
        print("Computing PCA...")
        features = None
        n_patch = 1
        for i, cam in enumerate(self.cameras):
            img = image_from_path(self.directory, cam.image_name, normalize=True)
            H, W = img.shape[1:]
            Hr, Wr = (H // 14) * 14, (W // 14) * 14
            img = nn.functional.interpolate(
                img[None], size=(Hr, Wr), mode="bilinear"
            ).squeeze()
            if self.sliding_window is not None:
                patches = self.sliding_window(img)
                patches = list(map(self.model.predict, patches))
                n_patch = len(patches)
                D, h, w = patches[0].shape
                flatten_fn = lambda x: x.view(D,-1) if self.use_cuda else x.view(D,-1).cpu().numpy()
                _features = list(map(flatten_fn, patches))
                _features = torch.hstack(_features).T
            else:
                feat = self.model.predict(img)
                D, h, w = feat.shape
                flatten_fn = lambda x: x.view(D,-1) if self.use_cuda else x.view(D,-1).cpu().numpy()
                _features = flatten_fn(feat).T
            npix = h*w*n_patch
            if features is None:
                if self.use_cuda:
                    features = torch.empty((npix*len(self.cameras), D), device='cuda', dtype=torch.float32)
                else:
                    features = np.empty((npix*len(self.cameras), D), dtype=np.float32)
            features[i*npix:(i+1)*npix] = _features
        features = self.apply_pca(features)
        split_indices = np.cumsum(npix * np.ones(len(self.cameras)).astype(int))
        self.features = np.split(features, split_indices[:-1], axis=0)
        self.n_patch = n_patch
        self.h, self.w = h, w
        self.H, self.W = H, W
        self.Hr, self.Wr = Hr, Wr

    def apply_pca(self, features):
        pca = PCA(n_components=self.n_components)
        if self.use_cuda:
            features -= features.mean(dim=0)
            features /= features.std(dim=0)
            pca_on = features
            if len(pca_on)>self.pca_subsample:
                indices = np.random.choice(range(len(pca_on)), self.pca_subsample, replace=False)
                pca_on = pca_on[indices]
            pca.fit(pca_on.cpu().numpy())
            features -= torch.from_numpy(pca.mean_).cuda()
            features = features @ torch.from_numpy(pca.components_.T).cuda()
            features = features.cpu().numpy()
        else:
            features -= features.mean(axis=0)
            features /= features.std(axis=0)
            features = pca.fit_transform(features)
        self.eigvals = torch.from_numpy(pca.singular_values_).cuda()
        return features
