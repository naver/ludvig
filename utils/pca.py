import torch
import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from .visualization import viz_normalization
from .image import save_img


def save_pca(pca, eval_dir, saturate=False, ext="", sgn_shift=None):
    if sgn_shift is None:
        sgn_shift = pca.mean(dim=(1, 2)) < 0
    pca[sgn_shift] *= -1
    pca = viz_normalization(pca, dim=(1, 2))
    if not saturate:
        save_img(
            os.path.join(eval_dir, f"pca{ext}.jpg"),
            pca,
        )
    else:
        rgb_image = np.transpose(pca.cpu().numpy(), (1, 2, 0))
        hsv_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.5, 0, 255)
        enhanced_rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        pca = np.transpose(enhanced_rgb_image / 255.0, (2, 0, 1))
        save_img(
            os.path.join(eval_dir, f"pca{ext}.jpg"),
            torch.from_numpy(pca).cuda(),
        )
    return sgn_shift


def save_pcas(pca, eval_dir, saturate=False, ext="", **kwargs):
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                l = int(i > 0) + int(j > 0) * 2 + int(k > 0) * 2**2
                _to_save = pca * torch.Tensor([i, j, k])[:, None, None].cuda()
                _to_save = viz_normalization(_to_save, dim=(1, 2))
                if ext and ext[0] == "/":
                    os.makedirs(os.path.join(eval_dir, f"pca_{l}"), exist_ok=True)
                if saturate:
                    rgb_image = np.transpose(_to_save.cpu().numpy(), (1, 2, 0))
                    hsv_image = cv2.cvtColor(
                        (rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV
                    )
                    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.5, 0, 255)
                    enhanced_rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
                    _to_save = np.transpose(enhanced_rgb_image / 255.0, (2, 0, 1))
                    save_img(
                        os.path.join(eval_dir, f"pca_{l}{ext}.jpg"),
                        torch.from_numpy(_to_save).cuda(),
                        **kwargs,
                    )
                else:
                    save_img(
                        os.path.join(eval_dir, f"pca_{l}{ext}.jpg"), _to_save, **kwargs
                    )


def pca_on_embeddings(
    dino_masks, n_components, pca=None, max_pred=500000, use_cuda=True, normalize=True
):
    original_mask_shapes = [
        mask.shape for mask in dino_masks
    ]  # Each shape is (d, h, w)
    d = original_mask_shapes[0][0]
    for i in range(len(dino_masks)):
        if use_cuda:
            dino_masks[i] = dino_masks[i].view(d, -1).T.cuda()
        else:
            dino_masks[i] = dino_masks[i].view(d, -1).T.cpu()
    dino_masks = torch.cat(dino_masks)  # (h*w*n_masks, d)

    if normalize:
        dino_masks -= dino_masks.mean(dim=0)
        dino_masks /= dino_masks.std(dim=0)

    # dino_masks = StandardScaler().fit_transform(dino_masks.cpu().numpy())
    if pca is None:
        pca = PCA(n_components=n_components)
        if len(dino_masks) > max_pred:
            print(f"Keeping {max_pred} patches out of {len(dino_masks)} for PCA.")
            sub_masks = np.random.choice(
                range(len(dino_masks)), max_pred, replace=False
            )
            pca.fit(
                dino_masks[sub_masks].cpu().numpy()
                if use_cuda
                else dino_masks[sub_masks]
            )
            print("Transforming masks...")
            if use_cuda:
                dino_masks -= torch.from_numpy(pca.mean_).cuda()
                dino_masks = dino_masks @ torch.from_numpy(pca.components_.T).cuda()
                dino_masks = dino_masks.cpu().numpy()
            else:
                dino_masks = pca.transform(dino_masks.cpu().numpy())
        else:
            dino_masks = pca.fit_transform(dino_masks.cpu().numpy())
    else:
        dino_masks = pca.transform(dino_masks)
    cumsizes = [0] + list(
        np.cumsum([s[1] * s[2] for s in original_mask_shapes])
    )  # (h*w*n_masks, d)
    dino_masks = [
        dino_masks[cumsizes[i] : cumsizes[i + 1]].T.reshape(
            (
                n_components,
                original_mask_shapes[i][1],
                original_mask_shapes[i][2],
            )
        )
        for i in range(len(original_mask_shapes))
    ]
    return dino_masks, pca
