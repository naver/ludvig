import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_scribbles(path, size, tags):
    scribbles = []
    for tag in tags:
        if bool(tag):
            scrib = next(f for f in os.listdir(path) if tag in f)
            img_path = os.path.join(path, scrib)
        else:
            img_path = path
        scrib = Image.open(img_path)
        scrib = np.array(scrib.resize(size, resample=Image.Resampling.NEAREST))
        scrib = torch.Tensor(scrib).cuda()
        scrib /= scrib.max()
        if len(scrib.shape) == 3:
            assert scrib.shape[-1] == 3
            scrib = scrib[..., 0]
        scribbles.append(scrib)
    return scribbles


def scribble_inverse_rendering(scribbles, gaussians, camera, return_counts=False):
    scribbles_3d = torch.zeros_like(gaussians._xyz)
    weights = torch.zeros_like(gaussians._opacity)
    scrib = scribbles[0]
    if len(scribbles) > 1:
        scrib -= scribbles[1]
    scrib = scrib[None].repeat(3, 1, 1)
    gaussians.apply_weights(camera, scribbles_3d, weights, scrib.cuda())
    scribbles_3d = scribbles_3d[:, :1] / (weights + 1e-8)
    if return_counts:
        return scribbles_3d, weights
    return scribbles_3d


def scribble_3d_similarities(features, weights, method="mmd"):
    if torch.any(weights < 0):
        eps = 1e-2
        wp = weights > 0
        wn = weights < 0
        wpz = (weights * wp).sum()
        wnz = (weights * wn).sum()

        if method == "lda":
            features = torch.nn.functional.normalize(features, dim=-1)
            anchor_p = ((features * weights) * wp / wpz).sum(0)
            anchor_n = ((features * weights) * wn / wnz).sum(0)

            # cov = torch.cov(features.T)
            # w = torch.linalg.solve(cov, anchor_p - anchor_n)
            # anchor = (features * w[None]).sum(dim=-1)
            # c = 0.5 * (w *(anchor_p + anchor_n)).sum()
            anchor = (features * (anchor_p - anchor_n)[None]).sum(dim=-1)
        else:
            wp, wn = torch.where(wp.squeeze()), torch.where(wn.squeeze())
            fp = features[wp]
            fn = features[wn]
            print(fp.shape, fn.shape)
            simp = torch.zeros_like(features[:, 0])
            print("Computing pairwise cosine similarities with positives.")
            for _fp, _wp in tqdm(zip(features[wp], weights[wp])):
                s = torch.nn.functional.cosine_similarity(features, _fp[None], dim=-1)
                simp += s * _wp / wpz
            simn = torch.zeros_like(features[:, 0])
            print("Computing pairwise cosine similarities with negatives.")
            for _fn, _wn in tqdm(zip(features[wn], weights[wn])):
                s = torch.nn.functional.cosine_similarity(features, _fn[None], dim=-1)
                simn += s * _wn / wnz
            anchor = simp - simn

            # kmeansp = KMeans(n_clusters=30, n_init="auto").fit(fp.cpu().numpy())
            # fp = torch.from_numpy(kmeansp.cluster_centers_).cuda()
            # kmeansn = KMeans(n_clusters=30, n_init="auto").fit(fn.cpu().numpy())
            # fn = torch.from_numpy(kmeansn.cluster_centers_).cuda()
            # simp = torch.nn.functional.cosine_similarity(features[None], fp[:,None], dim=-1)
            # simn = torch.nn.functional.cosine_similarity(features[None], fn[:,None], dim=-1)
            # anchor = torch.max(simp, dim=0)[0] - torch.max(simn, dim=0)[0]

        amin, amax = anchor.min(), anchor.max()
        anchor = (anchor - amin) / (amax - amin)
        anchor = anchor[:, None]

    else:
        wp = weights > 0
        wpz = (weights * wp).sum()
        wp = torch.where(wp.squeeze())
        fp = features[wp]
        anchor = torch.zeros_like(features[:, 0])
        print("Computing pairwise cosine similarities with positives.")
        for _fp, _wp in tqdm(zip(features[wp], weights[wp])):
            s = torch.nn.functional.cosine_similarity(features, _fp[None], dim=-1)
            anchor += s * _wp / wpz
        anchor = anchor[:, None]
        # fp = (features * weights / weights.sum()).sum(0, keepdim=True)
        # anchor = torch.nn.functional.cosine_similarity(features, fp, dim=-1)
        # amin, amax = anchor.min(), anchor.max()
        # anchor = (anchor - amin) / (amax - amin)
        # anchor = anchor[:, None]
    return anchor
