import os
import sys
import argparse
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
from torch.hub import load_state_dict_from_url

import swin_transformer

pretrained_weight_url = {
    'swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
    'swin_small': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth',
    'swin_base': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
    'swin_large': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
}

model_names = pretrained_weight_url.keys()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='swin_tiny',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: swin_tiny)')
    parser.add_argument('--patch-size', default=4, type=int, help='Patch resolution of the model. For swin transformer, that is 4.')
    parser.add_argument('--pretrained', default='', type=str,
        help="Path to pretrained weights to load.")
    # parser.add_argument("--checkpoint_key", default="teacher", type=str,
    #     help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--img-path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--img_size", default=224, type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output-dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Retrieve pretrained weights
    checkpoint=None
    if len(args.pretrained)==0:
        url=pretrained_weight_url[args.arch]
        downloadDir="pretrainedIN"
        checkpoint = load_state_dict_from_url(url, model_dir=downloadDir, map_location="cpu")
        print("Use downloaded pretrained weight at", os.path.join(downloadDir,os.path.basename(url)))
    else:
        assert os.path.isfile(args.pretrained), f"Given pretrained weights at {args.pretrained} does not exist."
        print(f"Use given pretrained weights at {args.pretrained}")
        checkpoint=torch.load(args.pretrained, map_location="cpu")
    
    # build model
    model=swin_transformer.__dict__[args.arch](state_dict=checkpoint["state_dict" if "state_dict" in checkpoint else "model"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # open image
    if args.img_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--img_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.img_path):
        with open(args.img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.img_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.img_size),
        pth_transforms.CenterCrop(args.img_size),
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
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, os.path.basename(args.img_path)))
    fname = os.path.join(args.output_dir, ("IN_" if len(args.pretrained)==0 else "SemCL_")+str(os.path.basename(args.img_path)).split(sep='.')[0]+"_attn-head" + ".png")
    plt.imsave(fname=fname, arr=attentions.squeeze(), format='png')
    print(f"{fname} saved.")
