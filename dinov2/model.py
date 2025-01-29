# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from functools import partial
from contextlib import nullcontext
from .setup import setup_and_build_model
from .eval_utils import ModelWithIntermediateLayers

class DINOv2(nn.Module):

    def __init__(self, config_file, pretrained_weights=None):
        super().__init__()
        model, autocast_dtype = setup_and_build_model(config_file, pretrained_weights)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.half)
        self.feature_model = model
        self.autocast_ctx = autocast_ctx
        #self.feature_model = ModelWithIntermediateLayers(model, n_last_blocks=4, autocast_ctx=autocast_ctx)
        for n, p in self.feature_model.named_parameters():
            p.requires_grad = False

    def forward(self, x):
        with self.autocast_ctx():
            x = self.feature_model.get_intermediate_layers(
                x, n=4, return_class_token=True, reshape=True
            )
        return x

    def predict(self, x):
        c, H, W = x.shape
        Hr, Wr = (H//14)*14, (W//14)*14
        x = x[None]
        if H!=Hr or W!=Wr:
            x = nn.functional.interpolate(input=x, size=(Hr,Wr), mode='bilinear')
        feat = self(x)[-1][0]
        return feat.squeeze()
