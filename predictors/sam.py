import os
import copy
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from utils.config import config_to_instance
from utils.data import fetch_data_info
from utils.image import image_from_path
from utils.scribble import load_scribbles, scribble_inverse_rendering
from utils.sam import load_sam
from .base import BaseDataset

class SAMDataset(BaseDataset):

    def __init__(
        self,
        *args,
        sam_ckpt,
        scribble,
        multimask_output,
        thres=0.4,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scribble = scribble.format(self.scene)
        self.model = load_sam(sam_ckpt)
        self.multimask_output = multimask_output
        self.thres = thres
        self.scribbles_3d = self.load_scribbles_3d()

    def load_scribbles_3d(self):
        _, _, _, scribble_info = fetch_data_info(self.scribble, self.cameras)
        cam_name = scribble_info.pop('cam_name')
        scribbles_2d = load_scribbles(**scribble_info, size=(self.width, self.height))
        camera = next(cam for cam in self.cameras if cam.image_name==cam_name)
        scribbles_3d = scribble_inverse_rendering(scribbles_2d, self.gaussian, camera)
        return scribbles_3d.repeat(1,3)

    @torch.no_grad()
    def __getitem__(self, idx):
        camera = self.cameras[idx]
        img = image_from_path(self.directory, camera.image_name)
        self.model.set_image(
            np.asarray(to_pil_image(img.cpu())).copy(),
        )
        _, height, width = img.shape
        pt2d = self.render_fn(self.scribbles_3d, camera)[0]
        npos = 3
        n_pred = 10
        sample_from = int(pt2d.sum() * self.thres)
        top_indices = np.argsort(
            pt2d.flatten().cpu().numpy()
        )[-sample_from:]
        top_indices = [
            np.random.choice(top_indices, npos, replace=False)
            for _ in range(n_pred)
        ]
        ppt = [
            np.array([(idx % width, idx // width) for idx in top_idx])
            for top_idx in top_indices
        ]
        point_labels = [1] * npos
        point_labels = np.array(point_labels, dtype=np.int64)
        mask = [
            self.model.predict(
                point_coords=_ppt,
                point_labels=point_labels,
                multimask_output=self.multimask_output,
                return_logits=False,
            )[0]
            for _ppt in ppt
        ]
        mask = sum(mask) / len(mask)
        mask /= mask.max()
        return torch.from_numpy(mask).to(torch.float32).cuda(), camera
