import os
import torch
from utils.image import save_img
from utils.config import config_to_instance
from evaluation.base import EvaluationBase


class Removal(EvaluationBase):

    def __init__(self, *args, thresholding=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        if self.features.shape[-1] == 1:
            self.features = self.features.repeat(1, 3)
        assert isinstance(thresholding, dict) or isinstance(
            thresholding, float
        ), f"Thresholding format not supported"
        self.to_numpy = False
        if isinstance(thresholding, dict):
            self.to_numpy = thresholding.pop("to_numpy", False)
            self.method = config_to_instance(**thresholding)
        else:
            self.method = lambda f: thresholding
        os.makedirs(os.path.join(self.logdir, "removal"), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, "rgb_masks"), exist_ok=True)

    def __call__(self):
        features = self.features[:, 0] / self.features.max()
        features_for_threshold = features
        if self.to_numpy:
            features_for_threshold = features.cpu().numpy()
        mask = features < self.method(features_for_threshold)
        for cam in self.colmap_cameras:
            mask_2d = self.render_fn(1-mask[:,None].type(torch.float32).repeat(1,3), cam)
            save_img(
                os.path.join(self.logdir, "masks", f"{cam.image_name}.jpg"), mask_2d
            )
            img = self.render_rgb(cam)["render"]
            save_img(
                os.path.join(self.logdir, "rgb_masks", f"{cam.image_name}.jpg"), img*mask_2d
            )
        self.gaussian.prune_points_noopt(
            torch.where(features < self.method(features_for_threshold))[0], backup=True
        )
        self.gaussian.save_ply(os.path.join(self.logdir, "removal.ply"))
        for cam in self.colmap_cameras:
            rgb_rm = self.render_rgb(cam)["render"]
            save_img(
                os.path.join(self.logdir, "removal", f"{cam.image_name}.jpg"), rgb_rm
            )
        self.gaussian.recover_points()
