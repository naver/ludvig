import os
import numpy as np
from tqdm import tqdm
from itertools import product
from .base import SegmentationBase
from diffusion.segmentation import GraphDiffusionSeg
from utils.image import save_img
from utils.visualization import viz_normalization


class SegmentationDiffusion(SegmentationBase):

    def __init__(self, *args, diffusion_cfg, gaussian, maxpos, **kwargs):
        super().__init__(*args, gaussian=gaussian, **kwargs)
        self.graph = GraphDiffusionSeg(
            gaussian,
            self.render_fn,
            self.colmap_cameras,
            self.logdir,
            scribbles_2d=self.scribbles_2d,
            scribble_camera=self.scribble_camera,
            maxpos=maxpos,
            **diffusion_cfg,
        )
        self.trace_name = self.graph.trace_name
        self.graph.trace_name = None
        self.manifold_features = None
        self.reg_similarities = None

    def bandwidth_hyperparameter_search(self, frange, grange, sam_as_gt=False):
        """Finds optimal bandwidth hyperparameters for graph diffusion."""

        k_best = 0
        f_best = 2
        g_best = 2
        results = []
        best_iou = 0
        param_combinations = list(product(frange, grange))
        bar_fmt = "{n_fmt}/{total_fmt} | Bandwidths: {postfix[0]}, {postfix[1]}  IoU: {postfix[2]:.3f} --  Best IoU: {postfix[3]:.3f}  Best bandwidths: {postfix[4]}, {postfix[5]}"
        with tqdm(
            total=len(param_combinations),
            bar_format=bar_fmt,
            postfix=["N/A", "N/A", 0.0, 0.0, "N/A", "N/A"],
        ) as pbar:
            for f, g in param_combinations:
                self.graph.feature_bandwidth = 2.0**f
                self.graph.reg_bandwidth = 2.0**g
                self.manifold_features, _ = self.graph(self.features)
                cur_iou, k_iou = self.segment_and_evaluate(
                    self.manifold_features,
                    save=False,
                    use_sam=self.sam_model is not None,
                )
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    k_best = k_iou
                    f_best = 2.0**f
                    g_best = 2.0**g
                pbar.postfix = [2.0**f, 2.0**g, cur_iou, best_iou, f_best, g_best]
                pbar.update(1)
                results.append((f, g, cur_iou))
        self.graph.feature_bandwidth = f_best
        self.graph.reg_bandwidth = g_best
        self.graph.trace_name = self.trace_name
        self.manifold_features, self.reg_similarities = self.graph(self.features)
        return k_best, f_best, g_best

    def evaluate(self, *args):
        if self.graph.initial_features is not None:
            save_img(
                os.path.join(self.logdir, "initial_points.png"),
                self.render_fn(
                    viz_normalization(self.graph.initial_features.repeat(1, 3)),
                    self.scribble_camera,
                ),
            )


class SegmentationDiffusionNVOS(SegmentationDiffusion):

    def hyperparameter_search(self):
        frange = np.arange(1, 3)
        grange = np.arange(2, 5)
        return self.bandwidth_hyperparameter_search(frange, grange, sam_as_gt=True)

    def evaluate(self, k_best, f_best, g_best):
        """Runs evaluation on NVOS with graph diffusion."""
        f_iou = open(os.path.join(self.logdir, "miou.txt"), "a")
        f_iou.write(f"Hyperparameters chosen based on IoU with SAM mask:\n")
        f_iou.write(f"\tSegmentation threshold: {k_best} \n")
        f_iou.write(f"\tFeature bandwidth: {f_best} \n")
        f_iou.write(f"\tReg bandwidth: {g_best} \n")
        best_iou, _ = self.segment_and_evaluate(
            self.manifold_features, save=True, k_best=k_best
        )
        f_iou.write(f"\nIoU: {round(best_iou,3)} \n")
        print(
            "Feature and regularization bandwiths, chosen based on IoU with SAM mask:",
            f_best,
            g_best,
        )
        print("IoU:", round(best_iou, 3))
        f_iou.close()
        super().evaluate()


class SegmentationDiffusionSPIn(SegmentationDiffusion):

    def hyperparameter_search(self):
        frange = np.arange(-1, 4)
        grange = np.arange(-3, 1)
        return self.bandwidth_hyperparameter_search(frange, grange, sam_as_gt=False)

    def evaluate(self, k_best, f_best=None, g_best=None):
        """Runs evaluation on SPin-NeRF using hyperparameters found with/without graph diffusion."""
        f_iou = open(os.path.join(self.logdir, "miou.txt"), "a")
        f_iou.write(f"Hyperparameters chosen based on IoU on reference view:\n")
        f_iou.write(f"\tSegmentation threshold: {k_best} \n")
        if f_best is not None:
            f_iou.write(f"\tFeature bandwidth: {f_best} \n")
            f_iou.write(f"\tReg bandwidth: {g_best} \n")
        res = []
        print(
            "Feature and regularization bandwiths, chosen based on IoU on reference view:",
            f_best,
            g_best,
        )
        for i, ev in enumerate(self.eval_paths):
            _iou, _ = self.segment_and_evaluate(
                self.manifold_features, k_best, ev_name=ev
            )
            if i > 0:
                res.append(_iou)
        print("Mean IoU:", round(sum(res) / len(res), 3))
        f_iou.write(f"\nMean IoU: {round(sum(res)/len(res), 3)}\n")
        f_iou.close()
        super().evaluate()
