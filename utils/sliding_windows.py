import torch
import numpy as np
from .image import resize


class SlidingWindow:
    def __init__(self, crop_size=50, stride=None, stride_diviser=None):
        assert type(crop_size) in [int, str]
        assert type(crop_size) == type(stride)
        self.crop_size = crop_size
        self.stride = stride or crop_size // 2
        self.stride_diviser = stride_diviser
        if isinstance(self.crop_size, str):
            self.crop_size = list(map(lambda x: int(x), self.crop_size.split(",")))
            self.stride = list(map(lambda x: int(x), self.stride.split(",")))
        else:
            self.crop_size = [self.crop_size]
            self.stride = [self.stride]
        self.indices = []

    def calculate_strides(self, stride, crop_size, d):
        if d == crop_size:
            return stride
        stride = int(np.ceil((d - crop_size) / np.ceil((d - crop_size) / stride)))
        if self.stride_diviser is not None:
            stride = int(np.ceil(stride / self.stride_diviser) * self.stride_diviser)
        return stride

    def __call__(self, img):
        h_img, w_img = img.shape[1:]
        patches = []
        sizes = []
        indices = []
        for stride, crop_size in zip(self.stride, self.crop_size):
            crop_size = min(min(crop_size, h_img), w_img)
            stride_h = self.calculate_strides(stride, crop_size, h_img)
            stride_w = self.calculate_strides(stride, crop_size, w_img)
            for y in range(0, h_img - crop_size + stride_h, stride_h):
                for x in range(0, w_img - crop_size + stride_w, stride_w):
                    y = min(y, h_img - crop_size)
                    x = min(x, w_img - crop_size)
                    patch = img[:, y : y + crop_size, x : x + crop_size]
                    patches.append(patch)
                    indices.append((y, x))
                    sizes.append(crop_size)
        self.indices.append(indices)
        self.sizes = sizes
        return patches

    def fill(self, patches, indices, size, predictor=None):
        result = None
        counts = None
        for i, (ind, patch) in enumerate(zip(indices, patches)):
            crop_size = self.sizes[i]
            if predictor is not None:
                with torch.no_grad():
                    patch = predictor(i, patch).squeeze()
            if result is None:
                result = torch.zeros(
                    (patch.shape[0], *size), device="cuda", dtype=torch.float32
                )
                counts = torch.zeros(size, device="cuda", dtype=torch.float32)
            result[:, ind[0] : ind[0] + crop_size, ind[1] : ind[1] + crop_size] += (
                patch / crop_size
            )
            counts[ind[0] : ind[0] + crop_size, ind[1] : ind[1] + crop_size] += (
                1 / crop_size
            )
        result /= counts[None]
        return result
