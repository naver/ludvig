import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from .evaluation import segmentation_2d
from .pca import pca_on_embeddings, save_pcas
from .config import config_to_instance
from .image import resize


def dino_evaluate(
    model,
    predict_fn,
    logreg_C,
    gt_img,
    eval_dir,
    scribbles_2d,
    rgb_img,
    rgb_img_scribbles=None,
    sliding_window=None,
    k_best=None,
):
    if sliding_window is not None:
        sliding_window = config_to_instance(**sliding_window)
    c, H, W = rgb_img.shape
    if sliding_window is not None:
        Hr, Wr = (H // 14) * 14, (W // 14) * 14
        img = F.interpolate(
            input=rgb_img[None], size=(Hr, Wr), mode="bilinear"
        ).squeeze()
        ftest = sliding_window(img, model.predict)
        c, h, w = ftest.shape
        pca, _ = pca_on_embeddings([ftest], 3, max_pred=100000)
        save_pcas(torch.from_numpy(pca[0]).cuda(), eval_dir)
    else:
        ftest = model.predict(rgb_img)
    if predict_fn is None:
        if sliding_window is not None:
            Hr, Wr = (H // 14) * 14, (W // 14) * 14
            img_scribbles = F.interpolate(
                input=rgb_img_scribbles[None], size=(Hr, Wr), mode="bilinear"
            ).squeeze()
            fscribbles = sliding_window(img_scribbles, model.predict)
        else:
            fscribbles = model.predict(rgb_img_scribbles)
        scribbles_2d = resize(scribbles_2d[None], tuple(fscribbles.shape[1:]))
        fscribbles = fscribbles.view(len(fscribbles), -1).T
        fscribbles /= fscribbles.norm(dim=1, keepdim=True) + 1e-8
        predict_fn = LogisticRegression(C=logreg_C, max_iter=1000).fit(
            fscribbles.cpu().numpy(),
            scribbles_2d.flatten().cpu().numpy().astype(int),
        )
    c, h, w = ftest.shape
    pca, _ = pca_on_embeddings([ftest], 3)
    save_pcas(torch.from_numpy(pca[0]).cuda(), eval_dir)
    ftest /= ftest.norm(dim=0, keepdim=True) + 1e-8
    anchor = predict_fn.predict_proba(ftest.view(len(ftest), -1).T.cpu().numpy())[:, 1]
    anchor = torch.from_numpy(anchor).view(h, w).cuda()
    anchor = resize(anchor[None], (H, W)).squeeze()
    best_iou, k_best = segmentation_2d(
        anchor, gt_img, rgb_img, eval_dir, k_best
    )
    return best_iou, k_best, predict_fn
