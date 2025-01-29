import os
import torch
from torch.utils.data import Dataset
from utils.image import image_from_path

class BaseDataset(Dataset):

    def __init__(
        self,
        directory,
        scene,
        gaussian,
        cameras,
        render_fn,
        height,
        width
    ):
        self.scene = scene
        self.directory = directory.format(self.scene)
        self.gaussian = gaussian
        self.cameras = cameras
        self.render_fn = render_fn
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.cameras)

    @torch.no_grad()
    def __getitem__(self, idx):
        cam = self.cameras[idx]
        return image_from_path(self.directory, cam.image_name), cam
