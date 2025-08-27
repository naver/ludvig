import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from skimage import filters
from torchvision.transforms.functional import to_pil_image
from .image import resize
from .evaluation import segmentation_2d


def load_sam2(ckpt_path):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model_cfg = "sam2_hiera_l.yaml"
    return SAM2ImagePredictor(build_sam2(model_cfg, ckpt_path))


def load_sam1(ckpt_path=None):
    sam_type = "vit_h"
    try:
        sam = sam_model_registry[sam_type](ckpt_path)
    except:
        raise ValueError(
            f"Problem loading SAM. Your model type: {sam_type} \
        should match your checkpoint path: {ckpt_path}."
        )
    sam.cuda()
    return SamPredictor(sam)


def load_sam(ckpt_path):
    return load_sam2(ckpt_path) if "sam2" in ckpt_path else load_sam1(ckpt_path)


def sam_predict(
    model,
    scribbles_2d,
    rgb_img=None,
    multimask_output=True,
    return_mask="li",
    npo=3,
    npr=10,
    **kwargs,
):
    positive_indices = torch.nonzero(scribbles_2d[0], as_tuple=False)[:, [1, 0]]
    if positive_indices.size(0) < npo*npr:
        repeats = (npo*npr + positive_indices.size(0) - 1) // positive_indices.size(0)
        positive_indices = positive_indices.repeat(repeats, 1)

    random_positive_indices = positive_indices[
        torch.randperm(positive_indices.size(0))[: npo * npr]
    ].reshape((npr, npo, 2))
    pred = sam_evaluate(
        model,
        rgb_img,
        pts2d=random_positive_indices,
        return_mask=return_mask,
        multimask_output=multimask_output,
        **kwargs,
    )
    return pred


def sam_evaluate(
    model,
    rgb_img,
    scribbles_2d=None,
    pts2d=None,
    return_mask=False,
    multimask_output=True,
    sam_index=None,
    gt_img=None,
    eval_dir=None,
    k_best=None,
):
    if pts2d is None:
        assert scribbles_2d is not None
        h, w = rgb_img.shape[-2:]
        pts2d = mask_to_point_prompt(scribbles_2d, top_n=5, size=(h, w))
    if rgb_img is not None:
        img = np.asarray(to_pil_image(rgb_img.cpu()))
        model.set_image(img)
    mask = [
        model.predict(
            point_coords=(
                pt2d.cpu().numpy() if isinstance(pt2d, torch.Tensor) else pt2d
            ),
            point_labels=np.array([1] * pt2d.shape[0], dtype=np.int64),
            multimask_output=multimask_output,
            return_logits=False,
        )[0]
        for pt2d in pts2d
    ]
    mask = sum(mask) / len(mask)
    if return_mask:
        assert return_mask in ["li", "otsu"]
        method = filters.threshold_li if return_mask == "li" else filters.threshold_otsu
        mask = mask.mean(0)[:, :, None]
        mask[mask < method(mask)] = 0
        return mask
    assert eval_dir is not None and gt_img is not None
    mask = torch.from_numpy(mask).cuda()
    if not multimask_output:
        sam_index = 0
    if sam_index is None and multimask_output:
        sam_res = [
            segmentation_2d(
                mask[i], gt_img, rgb_img, eval_dir, k_best_iou=k_best
            )
            for i in range(3)
        ]
        print("SAM results:", sam_res)
        sam_index = np.argmax([x[0] for x in sam_res])
        best_iou, k_best = sam_res[sam_index]
    else:
        best_iou, k_best = segmentation_2d(
            mask[sam_index], gt_img, rgb_img, eval_dir, k_best_iou=k_best
        )
    return best_iou, (k_best, sam_index)


def mask_to_point_prompt(scribbles_2d, thres=0.4, top_n=None, n_pred=100, size=None):
    height, width = scribbles_2d.shape
    if size is not None and height != size[0]:
        print("Resizing from {(height, width)} to {size}")
        scribbles_2d = resize(scribbles_2d.squueze(), size)
    sample_from = int(scribbles_2d.sum() * thres)
    top_indices = np.argsort(scribbles_2d.flatten().cpu().numpy())[-sample_from:]
    if top_n:
        top_indices = [
            np.random.choice(top_indices, top_n, replace=False) for _ in range(n_pred)
        ]
        pts2d = [
            np.array([(index % width, index // width) for index in top_idx])
            for top_idx in top_indices
        ]
    else:
        pts2d = np.array([(index % width, index // width) for index in top_indices])
    return pts2d
