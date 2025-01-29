import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from clip_utils import colormaps


def visualize_loc(image, point, bbox, save_path, prompt, fontsize=15):
    plt.figure()
    plt.imshow(image)
    rect = patches.Rectangle(
        (0, 0),
        image.shape[1] - 1,
        image.shape[0] - 1,
        linewidth=0,
        edgecolor="none",
        facecolor="white",
        alpha=0.3,
    )
    plt.gca().add_patch(rect)
    input_point = point.reshape(1, -1)
    input_label = np.array([1])
    show_points(input_point, input_label, plt.gca())
    show_box(bbox, plt.gca())
    plt.text(
        x=image.shape[1] - fontsize,
        y=fontsize,
        s=prompt,
        color="white",
        fontsize=fontsize,
        ha="right",
        va="top",
    )
    plt.axis("off")
    plt.savefig(save_path.format(prompt), bbox_inches="tight", pad_inches=0.0, dpi=200)
    plt.close()


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="firebrick",
        marker="o",
        s=marker_size,
        edgecolor="black",
        linewidth=2.5,
        alpha=1,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="o",
        s=marker_size,
        edgecolor="black",
        linewidth=1.5,
        alpha=1,
    )


def show_box(boxes, ax, color=None):
    if type(color) == str and color == "random":
        color = np.random.random(3)
    elif color is None:
        color = "black"
    for box in boxes.reshape(-1, 4):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                w,
                h,
                edgecolor=color,
                facecolor=(0, 0, 0, 0),
                lw=4,
                capstyle="round",
                joinstyle="round",
                linestyle="dotted",
            )
        )


def heatmap_fn(img, rel, mask=None):
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    composited_image = colormaps.apply_colormap(
        rel[:1].permute(1, 2, 0) / rel.max(), colormap_options
    ).permute(2, 0, 1)
    if mask is None:
        mask = rel < 0.3
    composited_image[mask] = img[mask] * 0.3
    return composited_image


def stack_mask(mask_base, mask_add):
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def load_gt_lerfdata(json_folder):
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), "frame_*.json")))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), "frame_*.jpg")))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, "r") as f:
            gt_data = json.load(f)

        h, w = gt_data["info"]["height"], gt_data["info"]["width"]
        name = gt_data["info"]["name"].split(".")[0]
        for prompt_data in gt_data["objects"]:
            label = prompt_data["category"]
            box = np.asarray(prompt_data["bbox"]).reshape(-1)  # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data["segmentation"])
            if img_ann[label].get("mask", None) is not None:
                mask = stack_mask(img_ann[label]["mask"], mask)
                img_ann[label]["bboxes"] = np.concatenate(
                    [img_ann[label]["bboxes"].reshape(-1, 4), box.reshape(-1, 4)],
                    axis=0,
                )
            else:
                img_ann[label]["bboxes"] = box
            img_ann[label]["mask"] = mask
        gt_ann[name] = img_ann
    return gt_ann, img_paths

def is_in_box(coord, bboxes):
    for box in bboxes.reshape(-1, 4):
        x1, y1, x2, y2 = box
        if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in coord):
            return True
    return False
