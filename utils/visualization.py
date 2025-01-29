import torch
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import auc


def mask_superposition(mask_arr, gt_img_arr):
    """Takes prediction and ground truth {0,1} valued masks."""
    mask_diff = mask_arr * gt_img_arr
    mask_diff = mask_diff[None].repeat(3, 0)
    mask_diff[1] += mask_arr * (1 - gt_img_arr)
    mask_diff[2] += gt_img_arr * (1 - mask_arr)
    mask_diff = (mask_diff >= 1).astype(float)
    mask_diff = torch.from_numpy(mask_diff).type(torch.float32)
    return mask_diff


def plot_roc(fpr_list, tpr_list, dst_dir, labels):
    if len(labels) > 1:
        cmap = plt.get_cmap("viridis")  # Choose any suitable colormap
        norm = mcolors.Normalize(vmin=labels[0], vmax=labels[-1])
        colors = [cmap(norm(label)) for label in labels]
    else:
        colors = ["green"]
    plt.figure()
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    for fpr, tpr, label, color in zip(fpr_list, tpr_list, labels, colors):
        roc_auc = auc(fpr, tpr)
        if len(labels) > 1:
            label = f"t: {label}, ROC: {roc_auc:.2f})"
        else:
            label = f"ROC: {roc_auc:.2f})"
        plt.plot(fpr, tpr, color=color, lw=2, label=label)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(dst_dir, "roc.png"))
    print("ROC:", round(roc_auc, 3))
    plt.close()


def viz_normalization(colors, dim=0, clip=3):
    if len(colors) > 1:
        colors = (colors - colors.mean(dim, keepdim=True)) / colors.std(
            dim, keepdim=True
        )
        if clip > 0:
            colors = colors.clip(-clip, clip)
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    return colors


def apply_colormap(tensor, colormap="winter"):
    """
    Apply a colormap to a tensor of values in the range [0, 1].

    Parameters:
    tensor (torch.Tensor): 1D tensor of values in the range [0, 1].
    colormap (str): Name of the colormap to apply (default: 'viridis').

    Returns:
    torch.Tensor: Tensor with shape (n, 3) where each value is mapped to RGB.
    """
    # Ensure tensor values are in the range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Get the colormap from matplotlib
    cmap = cm.get_cmap(colormap)

    # Apply colormap
    tensor_colored = torch.tensor(
        cmap(tensor.cpu().numpy())[:, :3], dtype=torch.float32
    ).cuda()  # Discard the alpha channel

    return tensor_colored
