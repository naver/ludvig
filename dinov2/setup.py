# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pathlib
from typing import Any, List, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.backends.cudnn as cudnn

from .models import build_model_from_cfg
from .dino_utils import load_pretrained_weights


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


def get_cfg_from_args(config_file):
    dinov2_default_config = load_config("configs/ssl_default_config")
    default_cfg = OmegaConf.create(dinov2_default_config)
    if isinstance(config_file, dict):
        cfg = OmegaConf.create(config_file)
    else:
        cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights=None):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights is not None:
        load_pretrained_weights(model, pretrained_weights, "teacher")
    else:
        print("No pretrained weights.")
    #model.eval()
    model.cuda()
    return model


def setup_and_build_model(config_file, pretrained_weights=None):
    cudnn.benchmark = True
    config = get_cfg_from_args(config_file)
    model = build_model_for_eval(config, pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype
