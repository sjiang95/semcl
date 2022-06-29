import argparse
from unicodedata import name
from functools import partial
from torchinfo import summary
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models

import swin_transformer
from swin_transformer import SwinTransformer

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))
model_names = ['swin_tiny', 'swin_small', 'swin_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
# deeplab
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16],
                    help="This option is valid for only resnet backbones.")

parser.add_argument('full_ckpt',default='',type=str, metavar='PATH',
                    help="Path to pretrained weights having same architecture with --arch option.")
parser.add_argument('--summary-only', action='store_true',
                    help='Print backbone summary without saving.')

def load_moco_backbone(backbone:nn.Module, linear_keyword,args):
    #load state_dict
    checkpoint = torch.load(args.full_ckpt, map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = backbone.load_state_dict(state_dict, strict=False)#
    assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}, f"Missing keys: {msg.missing_keys}"
    return backbone

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    par_dir=os.path.dirname(filename)
    if os.path.exists(par_dir) is not True:
        os.makedirs(par_dir)

    torch.save(state, filename)

def moco2bkb():
    # use this function to extract base encoder backbone from pretrained weights
    args = parser.parse_args()

    # Retrieve pretrained weights
    assert (len(args.full_ckpt)>0), "You have to specify pretrained ckpt path."
    # check existence of full ckpt
    assert (os.path.isfile(args.full_ckpt)), f"Given full checkpoint at {args.full_ckpt} does not exist."

    # This is valid for only resnet models
    if args.output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        # aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        # aspp_dilate = [6, 12, 18]
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('swin'):
        # Unlike moco whose pretrained weights contain both base and momentum encoder,
        # swin transformer pretrained weights contains only the backbone (base encoder) itself.
        model=swin_transformer.__dict__[args.arch](pretrained='')
        linear_keyword = 'head'
        model=load_moco_backbone(model,linear_keyword=linear_keyword,args=args)
    else:
        model = torchvision_models.__dict__[args.arch]( zero_init_residual=True,replace_stride_with_dilation=replace_stride_with_dilation)
        linear_keyword = 'fc'
        model=load_moco_backbone(model,linear_keyword=linear_keyword,args=args)

    summary(model,input_size=(1,3,224,224))
    if args.summary_only:
        print("In 'summary_only' mode, backbone will not be saved.")
    else:
        slash_idx=str(args.full_ckpt).rfind('/')
        bkb_filename=str(args.full_ckpt)[:slash_idx+1]+"bkb_"+str(args.full_ckpt)[slash_idx+1:]
        save_checkpoint(
            {'state_dict': model.state_dict(),}
            ,filename=bkb_filename)

if __name__ == '__main__':
    moco2bkb()