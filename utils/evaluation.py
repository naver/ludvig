import os
import torch
import numpy as np

from PIL import Image
from skimage import filters
from .image import resize, save_img
from .visualization import mask_superposition


def multi_thresholding(x, degree=2, method=None):
    if method is None:
        method = filters.threshold_otsu
    for i in range(degree):
        t = method(x)
        x = x[x > t]
    return t


def segmentation_loop(_img, gt_img, k_best_iou=-2, n=100, metric="iou"):
    assert metric in ["iou", "acc"]
    if metric != "iou":
        print("Using accuracy metric.")
    best_iou = 0
    mask_best_iou = None
    fpr_list, tpr_list = [], []
    if isinstance(k_best_iou, str):
        assert k_best_iou == "li", "Only Li thresholding is supported"
        auto_thres = filters.threshold_li(_img.cpu().numpy())
        loop_range = [(1 - auto_thres) * 100]
    else:
        loop_range = range(1, n - 2) if k_best_iou is None else [k_best_iou]
    for k in loop_range:
        mask_2d = _img > (1 - k / 100)
        img = to_pil(mask_2d)
        img_arr, gt_img_arr = np.array(img), np.array(gt_img)
        img_arr = img_arr // max(img_arr.max(), 1)
        gt_img_arr = gt_img_arr // gt_img_arr.max()
        if metric == "iou":
            _iou = iou(img_arr, gt_img_arr, class_label=1)
        else:
            _iou = f2_score(gt_img_arr, img_arr, class_label=1)
        if _iou > best_iou:
            best_iou = _iou
            k_best_iou = k
            mask_best_iou = img
        tpr, fpr = tpr_fpr(img_arr, gt_img_arr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    fpr_list = [0] + fpr_list + [1]
    tpr_list = [0] + tpr_list + [1]
    # best_iou = int(round(100*best_iou))
    return (
        best_iou,
        k_best_iou,
        mask_best_iou,
        [fpr_list, tpr_list],
    )


def segmentation_2d(img, gt_img, rgb_img, eval_dir, k_best_iou):
    os.makedirs(eval_dir, exist_ok=True)
    gt_img.save(os.path.join(eval_dir, "gt.png"))
    _img = (img - img.min()) / (img.max() - img.min())
    _img_up = resize(_img[None], (gt_img.size[1], gt_img.size[0])).squeeze()
    best_iou, k_best_iou, mask_best_iou, fpr_tpr = segmentation_loop(
        _img_up, gt_img, k_best_iou=k_best_iou
    )
    text = "Cosine similarities (top)\nand mask (bottom)\n"
    text += f"IoU: {best_iou}"
    rgb_mask = rgb_img * (_img > 1 - k_best_iou / 100).cuda()
    save_img(os.path.join(eval_dir, f"masks/rgb/iou{round(100*best_iou)}.png"), rgb_mask)
    save_img(
        os.path.join(eval_dir, f"masks/comp/iou{round(100*best_iou)}.png"),
        _img,
        mask_superposition(np.array(mask_best_iou), np.array(gt_img)),
        text=text,
    )
    save_img(os.path.join(eval_dir, f"masks/float/iou{round(100*best_iou)}.png"), _img)
    save_img(os.path.join(eval_dir, f"masks/binary/iou{round(100*best_iou)}.png"), mask_best_iou)
    # fpr_list, tpr_list = fpr_tpr
    # plot_roc([fpr_list], [tpr_list], eval_dir, [None])
    return best_iou, k_best_iou


def iou_sa3d(a, b):
    """Calculates the Intersection over Union (IoU) between two ndarrays.

    Args:
        a: shape (N, H, W).
        b: shape (N, H, W).

    Returns:
        Shape (N,) containing the IoU score between each pair of
        elements in a and b.
    """
    intersection = np.count_nonzero(np.logical_and(a == b, a != 0))
    union = np.count_nonzero(a + b)
    return intersection / union


def f2_score(y_true, y_pred, class_label):
    true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
    false_positives = np.sum((y_true != class_label) & (y_pred == class_label))
    false_negatives = np.sum((y_true == class_label) & (y_pred != class_label))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f2 = (5 * precision * recall) / (4 * precision + recall)
    return f2


def acc(y_true, y_pred, class_label):
    return np.sum((y_true == y_pred)) / len(y_true.flatten())


def miou(y_true, y_pred, void_id):
    classes = np.unique(y_true)
    ious = []
    for cls in classes:
        if void_id is not None and cls == void_id:
            continue
        ious.append(iou(y_true, y_pred, cls))
    return sum(ious) / len(ious), {k: iou for k, iou in zip(classes, ious)}


def iou(y_true, y_pred, class_label):
    intersection = np.sum((y_true == class_label) & (y_pred == class_label))
    union = np.sum((y_true == class_label) | (y_pred == class_label))
    if union == 0:
        return 1.0  # If there is no ground truth mask and no predicted mask, consider it a perfect match.
    else:
        return intersection / union


def viz_normalization(colors, dim=0, clip=True):
    if len(colors) > 1:
        colors = (colors - colors.mean(dim, keepdim=True)) / colors.std(
            dim, keepdim=True
        )
        if clip:
            colors = colors.clip(-3, 3)
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    return colors


def tpr_fpr(img1, img2):
    fp = np.sum(img1 * (1 - img2))
    tp = np.sum(img1 * img2)
    fn = np.sum((1 - img1) * img2)
    tn = np.sum((1 - img1) * (1 - img2))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr


def to_pil(img):
    img = img.type(torch.float32)
    img = (img - img.min()) / (img.max() - img.min())
    if len(img.shape) == 3 and len(img) == 3:
        img = img.moveaxis(0, -1)
    img = (img * 255).to(torch.uint8).cpu()
    img = img.numpy().astype(np.uint8)
    img = Image.fromarray(img)
    return img
