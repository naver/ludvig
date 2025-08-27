import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.image import image_from_path
from .base import BaseDataset

class ScanNetDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory = self.directory.replace('images', 'language_features')

    def __getitem__(self, idx):
        s_map = np.load(os.path.join(self.directory, self.cameras[idx].image_name+'_s.npy'))[0]
        f_map = np.load(os.path.join(self.directory, self.cameras[idx].image_name+'_f.npy'))
        s_map = torch.from_numpy(s_map).long().cuda()
        f_map = torch.from_numpy(f_map).type(torch.float32).cuda()
        f_map /= f_map.norm(dim=1, keepdim=True) + 1e-8

        H, W = s_map.shape
        K, D = f_map.shape

        valid_mask = s_map != -1  # (H, W)
        flat_valid_indices = s_map[valid_mask]  # shape (N,), values in [0, K-1]
        selected_features = torch.index_select(f_map, 0, flat_valid_indices)  # (N, 512)
        output = torch.zeros(H, W, D, dtype=f_map.dtype, device=f_map.device)  # (H, W, 512)
        output[valid_mask] = selected_features  # (N, 512) goes into selected (H, W, 512) locations

        return output.permute(2,0,1), self.cameras[idx]
