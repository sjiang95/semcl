# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import torch


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]

class TwoCropsTransformWithItself:
    """Take two random crops of one image"""

    def __init__(self, base_transform0, base_transform1, base_transform2):
        self.base_transform0 = base_transform0
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x0, x1):
        im0_0, im0_1 = self.base_transform0(x0, x1)
        im1_0, im1_1 = self.base_transform1(x0, x1)
        im2_0, im2_1 = self.base_transform2(x0, x1)
        return torch.stack([im0_0, im1_0, im2_0]), torch.stack([im0_1, im1_1, im2_1])

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)