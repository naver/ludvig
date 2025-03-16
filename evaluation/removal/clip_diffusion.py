import os
import torch
from .base import Removal
from utils.image import save_img
from diffusion.clip import GraphDiffusionCLIP
from clip_utils.openclip_encoder import OpenCLIPNetwork
from clip_utils.visualization import heatmap_fn
from skimage import filters

class CLIPDiffusionRemoval(Removal):

    def __init__(self, *args, diffusion_cfg, gaussian, prompt, **kwargs):
        super().__init__(*args, gaussian=gaussian, **kwargs)
        self.clip_model = OpenCLIPNetwork("cuda")
        self.clip_model.set_positives([prompt])
        self.features /= torch.norm(self.features, dim=-1, keepdim=True) + 1e-6
        relev = self.clip_model.get_max_across(self.features)
        self.graph = GraphDiffusionCLIP(
            gaussian = gaussian,
            render_fn = self.render_fn,
            cameras = self.colmap_cameras,
            logdir = self.logdir,
            relev = relev,
            load_dino = os.path.join(self.logdir, "dinov2", "features.npy"),
            eps = 1e-6,
            **diffusion_cfg,
        )
        self.features = self.graph().squeeze()[:, None].repeat(1,3)

        ########## Visualizations ##########
        cam = self.colmap_cameras[0]
        relev_2d = self.render_fn(relev.T.repeat(1,3), cam)
        relev_2d = heatmap_fn(
            self.render_rgb(cam)["render"],
            relev_2d,
            mask = relev_2d < .3
        )
        save_img(os.path.join(self.logdir, f"localization.jpg"), relev_2d)
        initial_features = self.render_fn(
            self.graph.initial_features.max(dim=1, keepdim=True).values.repeat(1, 3),
            cam
        )
        regularizer = self.graph.compute_regularizer().max(dim=1, keepdim=True).values.repeat(1, 3)
        regularizer = self.render_fn(regularizer, cam)
        regularizer = heatmap_fn(
            self.render_rgb(cam)["render"],
            regularizer,
            mask = regularizer < filters.threshold_li(regularizer[0].cpu().numpy().flatten())
        )
        save_img(os.path.join(self.logdir, f"regularizer_{self.graph.reg_bandwidth}.jpg"), regularizer)
        save_img(os.path.join(self.logdir, f"initial_features.jpg"), initial_features)
