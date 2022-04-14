# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import swin_transformer

pretrained_weight_url={
    'swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth', 
    'swin_small': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth', 
    'swin_base': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
}

model_names = ['swin_tiny', 'swin_small', 'swin_base']

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

def load_moco_backbone(backbone:nn.Module, linear_keyword,args):
    #load state_dict
    checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    backbone.load_state_dict(state_dict, strict=False)#
    # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
    return backbone

def download_preweights(list_url, download_path, key):
    print("Download pretrained weights for %s backbone from '%s'." % (key,list_url[key]))
    down_res=requests.get(list_url[key])
    with open(download_path,'wb') as file:
        file.write(down_res.content)
    print("Download pretrained weights for %s backbone is saved to '%s'." % (key,download_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='swin_tiny',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: swin_tiny)')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch resolution of the model. For swin transformer, that is 4.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    # parser.add_argument("--checkpoint_key", default="teacher", type=str,
    #     help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Retrieve pretrained weights
    if len(args.pretrained_weights)==0:
        pretrained_weights_filename=pretrained_weight_url.copy()
        for one_key,one_value in pretrained_weights_filename.items():
            pretrained_weights_filename[one_key]=str(one_value).split(sep='/')[-1]

        path_to_pretrained_weights=os.path.join('pretrained',pretrained_weights_filename[args.arch])
        if not os.path.exists('pretrained'):
            os.mkdir('pretrained')
            download_preweights(pretrained_weight_url,path_to_pretrained_weights,args.arch)
        else:
            if not os.path.exists(path_to_pretrained_weights): # Download dict file if not exists
                download_preweights(pretrained_weight_url,path_to_pretrained_weights,args.arch)
        args.pretrained_weights=path_to_pretrained_weights
        print("Use downloaded pretrained weight at", args.pretrained_weights)
        linear_keyword = ''
    else:
        assert os.path.isfile(args.pretrained_weights), f"Given pretrained weights at {args.pretrained_weights} does not exist."
        print(f"Use given pretrained weights at {args.pretrained_weights}")
        linear_keyword = 'head'
    
    # build model
    model=swin_transformer.__dict__[args.arch](pretrained=('' if len(args.pretrained_weights)>0 else args.pretrained_weights))
    if len(linear_keyword)>0:
        model=load_moco_backbone(model,linear_keyword=linear_keyword,args=args)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.patch_embed(img.to(device))
    if model.ape:
        attentions = attentions + model.absolute_pos_embed
    attentions = model.pos_drop(attentions)
    attentions = attentions.transpose(1, 2).sum(dim=1,keepdim=True).reshape(1,56,56)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), size=(224,224), mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    args.output_dir=os.path.join(args.output_dir,"visualize_attention")
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, os.path.basename(args.image_path)))
    fname = os.path.join(args.output_dir, ("semcl_" if len(linear_keyword)>0 else "imagenet_")+str(os.path.basename(args.image_path)).split(sep='.')[0]+"_attn-head" + ".png")
    plt.imsave(fname=fname, arr=attentions.squeeze(), format='png')
    print(f"{fname} saved.")
