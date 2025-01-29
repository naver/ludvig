import os
import numpy as np
from .base import SegmentationBase


class SegmentationNVOSSAM(SegmentationBase):

    def __call__(self):
        """Runs evaluation on NVOS (no graph diffusion)."""
        eval_kwargs = dict(
            features=self.features.mean(dim=1, keepdim=True), normalize=True
        )
        _best_iou_th, k_best_th = self.segment_and_evaluate(
            save=False, k_best=None, **eval_kwargs
        )
        f_iou = open(os.path.join(self.logdir, "miou.txt"), "a")
        print(
            f"IoU for best threshold ({round(k_best_th, 1)}):",
            round(_best_iou_th, 3),
        )
        f_iou.write(f"IoU for best threshold ({k_best_th}): {round(_best_iou_th, 3)}\n")
        _best_iou, k_best = self.segment_and_evaluate(
            save=True, k_best=self.thresholding, **eval_kwargs
        )
        print(
            f"IoU for selected threshold ({round(k_best, 1)}):",
            round(_best_iou, 3),
        )
        f_iou.write(
            f"IoU for selected threshold ({round(k_best, 1)}): {round(_best_iou, 3)}\n"
        )


class SegmentationNVOSDINOv2(SegmentationBase):
    def __call__(self):
        """Runs evaluation on NVOS (no graph diffusion)."""
        assert self.sam_model is not None
        sam_iou, k_best = self.segment_and_evaluate(
            save=True, use_sam=True, normalize=False
        )
        _best_iou, _ = self.segment_and_evaluate(
            save=True, normalize=False, k_best=k_best
        )
        print(f"Threshold maximizing IoU with SAM ({sam_iou}) mask prediction: {k_best}")
        print(f"IoU for threshold {k_best}:", round(_best_iou, 3))


class SegmentationSPIn(SegmentationBase):
    def evaluate_spin(self, model, k_best, sid_best_iou=None):
        """Runs evaluation on SPin-NeRF using hyperparameters found with/without graph diffusion."""
        sam_index = None
        features = self.features
        if model == "dinov2":
            features = None
            if self.single_view:
                sam_index = sid_best_iou
        if model == "sam" and not self.single_view:
            features = features[:, sid_best_iou : sid_best_iou + 1]
        f_iou = open(os.path.join(self.logdir, "miou.txt"), "a")
        f_iou.write(f"Hyperparameters chosen based on IoU on reference view:\n")
        f_iou.write(f"\tSegmentation threshold: {k_best} \n")
        if sid_best_iou is not None:
            f_iou.write(
                f"\tSAM mask index: {sid_best_iou} \n"
            )
        else:
            sid_best_iou = 0
        res = []
        for i, ev in enumerate(self.eval_paths):
            _iou, *args = self.segment_and_evaluate(
                k_best=k_best,
                features=features,
                ev_name=ev,
                sam_index=sam_index,
                normalize=True,
            )
            if i > 0:
                res.append(_iou)
        print("Mean IoU:", round(sum(res) / len(res), 3))
        f_iou.write(f"Mean IoU: {round(sum(res)/len(res), 3)} \n")
        f_iou.close()


class SegmentationSPInSAM(SegmentationSPIn):

    def hyperparameter_search(self):
        """Finds optimal hyperparameters for SPIn-NeRF (no graph diffusion)."""
        if self.single_view:
            _, (k_best, sid_best_iou) = self.segment_and_evaluate(save=False)
        else:
            sam_res = [
                self.segment_and_evaluate(
                    features=self.features[:, i : i + 1], save=False, normalize=True
                )[:2]
                for i in range(3)
            ]
            print(
                "IoU per SAM mask on reference view:",
                [round(x[0], 3).item() for x in sam_res],
            )
            sid_best_iou = np.argmax([x[0] for x in sam_res])
            k_best = sam_res[sid_best_iou][1]
        print(
            f"Using threshold {k_best} and SAM mask index {sid_best_iou} based on IoU on reference view."
        )
        return k_best, sid_best_iou

    def evaluate(self, k_best, sid_best_iou=None):
        return self.evaluate_spin("sam", k_best, sid_best_iou)


class SegmentationSPInDINOv2(SegmentationSPIn):

    def hyperparameter_search(self):
        """Finds optimal hyperparameters for SPIn-NeRF (no graph diffusion)."""
        _best_iou, k_best = self.segment_and_evaluate(save=False, normalize=True)
        return k_best, None

    def evaluate(self, k_best, sid_best_iou=None):
        return self.evaluate_spin("dinov2", k_best, sid_best_iou)
