import sys
import numpy as np
import torch
import os
import yaml
from time import time
from argparse import ArgumentParser
from sklearn.decomposition import PCA

from utils.solver import uplifting
from utils.config import config_to_instance
from ludvig_base import LUDVIGBase, reproducibility


class LUDVIGUplift(LUDVIGBase):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.features = None
        load_ply = self.config.get('load_ply', None)
        if 'load_ply' in cfg and cfg.load_ply:
            load_ply = cfg.load_ply
        if load_ply:
            print("Loading gaussians from", load_ply)
            self.gaussian.load_ply(
                os.path.join(self.config.dst_dir, self.scene, load_ply)
            )
        self.save_visualizations = 'save_visualizations' in cfg and cfg.save_visualizations

    def uplift(self):
        """Initialize the dataset and uplift the feature maps it generates."""
        if self.config.get("feature", None) is None:
            return
        elif isinstance(self.config.feature, str):
            feature_path = os.path.join(self.config.dst_dir, self.scene, self.config.feature)
            print("Loading features from", feature_path)
            self.features = torch.from_numpy(np.load(feature_path)).cuda()
            return self.features
        t0 = time()
        print("Uplifting features...")
        directory = self.config['feature'].pop(
            'directory',
            os.path.join(self.colmap_dir, 'images')
        )
        dataset = config_to_instance(
            directory=directory,
            gaussian=self.gaussian,
            cameras=self.colmap_cameras,
            render_fn=self.render,
            scene=self.scene,
            height=self.img_height,
            width=self.img_width,
            **self.config.feature,
        )
        loader = iter(dataset)
        features, _ = uplifting(
            loader,
            self.gaussian,
            prune_gaussians=self.config.get("prune_gaussians", None),
        )
        if self.config.get('normalize', False):
            print("l2-normalizing uplifted features.")
            features /= features.norm(dim=1, keepdim=True) + 1e-6
        print(
            f"Total time for preprocessing + uplifting {len(self.colmap_cameras)} images: {round(time()-t0)}s"
        )
        self.features = features
        return features

    def save(self):
        """Save features and visualizations or run evaluation if specified in configuration."""
        os.makedirs(self.logdir, exist_ok=True)
        cfg_path = os.path.join(self.logdir, "config.yaml")
        yaml.dump(self.config, open(cfg_path, "w"))
        eval_kwargs = self.config.get("evaluation", dict())
        if "name" in eval_kwargs:
            eval_fn = config_to_instance(
                gaussian=self.gaussian,
                features=self.features,
                render_fn=self.render,
                render_rgb=self.render_rgb,
                logdir=self.logdir,
                image_dir=self.colmap_dir,
                colmap_cameras=self.colmap_cameras,
                scene=self.scene,
                height=self.img_height,
                width=self.img_width,
                **self.config.evaluation,
            )
            eval_fn()
        else:
            features = self.features
            if self.save_visualizations:
                if self.config.get('apply_pca', False):
                    print("Applying PCA to uplifted features for visualization.")
                    features = torch.from_numpy(
                        PCA(n_components=3).fit_transform(self.features.cpu().numpy())
                    ).cuda()
                self.save_images(features, pca=self.features.shape[1]>3, **eval_kwargs)
        self.gaussian.save_ply(os.path.join(self.logdir, "gaussians.ply"))
        np.save(os.path.join(self.logdir, "features.npy"), self.features.cpu().numpy())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--load_ply", type=str)
    parser.add_argument("--save_visualizations", action='store_true')
    parser.add_argument("--height", type=int, default=1199)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--tag", type=str)  #

    args = parser.parse_args()
    reproducibility(0)
    model = LUDVIGUplift(args)
    model.uplift()
    model.save()
    sys.exit()
