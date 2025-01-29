import os
import numpy as np
import torch
import torch.nn.functional as F
from clip_utils.openclip_encoder import OpenCLIPNetwork
from utils.image import image_from_path
from .base import BaseDataset

class SlidingWindow:
    def __init__(self, tile_ratios=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], stride_ratios=None):
        self.tile_ratios = tile_ratios
        self.stride_ratios = self._stride_scaler(self.tile_ratios) if stride_ratios is None else stride_ratios
        self.clip_dim = 512
        self.device = 'cuda'

    def _stride_scaler(self, tile_ratio):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, 0.5])

    def calculate_strides(self, stride, crop_size, d):
        if d == crop_size:
            return stride
        stride = int(np.ceil((d - crop_size) / np.ceil((d - crop_size) / stride)))
        if self.stride_diviser is not None:
            stride = int(np.ceil(stride / self.stride_diviser) * self.stride_diviser)
        return stride

    def __call__(self, image, model):
        h_img, w_img = image.shape[1:]
        multiscale_clip_embed = torch.zeros(
            (self.clip_dim, h_img, w_img),
            device="cuda",
            dtype=torch.float32,
        )
        for tile_ratio, stride_ratio in zip(self.tile_ratios, self.stride_ratios):
            kernel_size = int(tile_ratio * h_img)
            stride_h = stride_w = int(kernel_size * stride_ratio)
            clip_embeds = self.tile_and_embed(image, model, kernel_size, stride_h, stride_w) # ()
            clip_embeds = clip_embeds.permute(2, 0, 1)[None] # (1, 512, n_x + 1, n_y + 1)
            multiscale_clip_embed += F.interpolate(clip_embeds, (h_img, w_img), mode='bilinear').squeeze()
        multiscale_clip_embed /= multiscale_clip_embed.norm(dim=0, keepdim=True)

        return multiscale_clip_embed

    def tile_and_embed(self, image, model, kernel_size, stride_h, stride_w):
        stride = stride_h # Same stride in both directions to be consistent with LERF
        # number of rows and columns in the tiling
        padding = kernel_size // 2
        n_x = int(np.floor((image.shape[1] + 2 * padding - (kernel_size - 1) - 1) / stride + 1))
        n_y = int(np.floor((image.shape[2] + 2 * padding - (kernel_size - 1) - 1) / stride + 1))
        unfold_func = torch.nn.Unfold(
            kernel_size=kernel_size,
            stride=stride_h,
            padding=padding,
        ).to(self.device)

        tiles = unfold_func(image[None]).permute(2, 0, 1)
        tiles = tiles.reshape(-1, 3, kernel_size, kernel_size).to("cuda")
        with torch.no_grad():
            clip_embeds = model.encode_image(tiles)
        clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

        clip_embeds = clip_embeds.reshape((n_x, n_y, -1)) # (n_x, n_y, 512)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1) # (n_x, n_y + 1, D)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0) # (n_x + 1, n_y + 1, D)

        return clip_embeds


class CLIPDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tile_ratios = np.linspace(0.05, 0.5, 7)
        stride_ratios = np.interp(tile_ratios, [0.05, 0.15], [1.0, 0.5])
        self.sliding_window = SlidingWindow(tile_ratios=tile_ratios, stride_ratios=stride_ratios)
        self.model = OpenCLIPNetwork('cuda')

    def __getitem__(self, idx):
        img = image_from_path(self.directory, self.cameras[idx].image_name)
        h, w = img.shape[1:]
        feature_map = self.sliding_window(img, self.model)
        return feature_map.to(torch.float32), self.cameras[idx]
