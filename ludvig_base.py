import os
import numpy as np
import torch
import random
from argparse import ArgumentParser

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.camera_scene import CamScene
from gaussiansplatting.scene import Scene
from gaussiansplatting.arguments import PipelineParams

from utils.image import save_img
from utils.config import Config, config_to_instance
from utils.camera import interpolate_cameras
from utils.pca import save_pcas


def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LUDVIGBase:
    def __init__(self, cfg) -> None:
        self.config = Config(cfg.config)
        tag = self.config.get("tag", "")
        if 'tag' in cfg and cfg.tag:
            tag = "/".join((cfg.tag, tag))
        assert tag, "No tag provided."
        self.scene = tag.split("/")[0]
        self.logdir = os.path.join(self.config.dst_dir, tag)
        self.init_gaussians(cfg)

    def render_rgb(self, cam):
        """Render RGB image from camera."""
        return render(cam, self.gaussian, self.pipe, self.background_tensor)

    def init_gaussians(self, cfg):
        """Load cameras and Gaussian Splatting scene representation."""
        self.colmap_dir = cfg.colmap_dir
        self.img_height = cfg.height
        self.img_width = cfg.width

        self.gaussian = GaussianModel(sh_degree=0)
        gs_source = cfg.gs_source
        if os.path.isdir(gs_source):
            ply_file = next(f for f in os.listdir(gs_source))
            gs_source = os.path.join(gs_source, ply_file)
        self.colmap_cameras = None
        self.render_cameras = None

        if self.colmap_dir is not None:
            if "sparse" in os.listdir(self.colmap_dir):
                self.gaussian.load_ply(gs_source)
                scene = CamScene(self.colmap_dir, h=self.img_height, w=self.img_width)
                self.cameras_extent = scene.cameras_extent
                self.colmap_cameras = scene.cameras
            else:
                dataset = Config(dict(
                    model_path = self.logdir,
                    source_path = self.colmap_dir,
                    white_background = True,
                    eval = True,
                    resolution = 1,
                    data_device = "cuda",
                ))
                scene = Scene(dataset, self.gaussian)
                self.colmap_cameras = scene.getTrainCameras()
                self.gaussian.load_ply(gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        print(len(self.gaussian._xyz), "Gaussians.")
        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        # self.background_tensor = torch.rand(3, device="cuda")
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

    @torch.no_grad()
    def render(self, features, cam):
        """Render 2D feature map (D, H, W) based on 3D feature (N, D) and camera."""
        n_feat = features.shape[-1]
        counts = torch.zeros(n_feat, dtype=torch.float32, device="cuda")
        sem = None
        for j in np.arange(0, n_feat, 3):
            _j = min(j, n_feat - 3)
            _j = max(_j, 0)
            _semantic_map = render(
                cam,
                self.gaussian,
                self.pipe,
                self.background_tensor,
                override_color=features[:, _j : _j + 3],
            )["render"]
            if sem is None:
                sem = torch.zeros(
                    (n_feat, *_semantic_map.shape[1:]),
                    dtype=torch.float32,
                    device="cuda",
                )
            dj = min(3, len(sem) - _j)
            sem[_j : _j + dj] += _semantic_map[:dj]
            counts[_j : _j + dj] += 1
        sem /= counts[:, None, None]
        return sem

    def save_images(
        self,
        features,
        name="",
        joint_fn=None,
        cameras=None,
        pca=False,
        saturate_pca=False,
        interpolate=0,
        **kwargs,
    ):
        """
        Renders features based on provided cameras and saves visualizations,
        with optional interpolation to generate videos.

        Args:
            features (torch.Tensor): 3D feature to render, with shape (N, D).
            name (str, optional): Name for the subdirectory where images will be saved.
            joint_fn (callable, optional): A function that combines RGB images and feature maps
                                           into a single image for visualizations.
            cameras (list, optional): A list of cameras to render from.
                                      Defaults to `self.colmap_cameras` if not provided.
            pca (bool, optional): Whether the features are PCA-transformed.
                                  If true, multiple color variants are saved.
            saturate_pca (bool, optional): Whether to saturate colors in the PCA visualization
            interpolate (int, optional): Number of interpolated cameras to generate between the given
                                         cameras (used for videos). If 0, no interpolation is performed.
        """
        if cameras is None:
            cameras = self.colmap_cameras
        if interpolate:
            cameras = interpolate_cameras(
                cameras, smoothness=3, n_interp=interpolate, spline_degree=2
            )
        save_dir = os.path.join(self.logdir, "features", name)
        os.makedirs(save_dir, exist_ok=True)
        if joint_fn is not None and isinstance(joint_fn, dict):
            joint_fn = config_to_instance(**joint_fn)
        for i, camera in enumerate(cameras):
            istr = str(i).zfill(3)
            rgb_image = self.render_rgb(camera)["render"]
            feat_img = self.render(features, camera)
            if joint_fn is not None:
                image = joint_fn(rgb_image, feat_img)
                save_img(os.path.join(save_dir, f"{istr}.jpg"), image, **kwargs)
            elif pca:
                os.makedirs(os.path.join(save_dir, f"feat_{istr}"), exist_ok=True)
                save_pcas(
                    feat_img[:3], os.path.join(save_dir, f"feat_{istr}"), saturate=saturate_pca
                )
            else:
                save_img(os.path.join(save_dir, f"rgb_{istr}.jpg"), rgb_image)
                save_img(os.path.join(save_dir, f"feat_{istr}.jpg"), feat_img[:3], **kwargs)
