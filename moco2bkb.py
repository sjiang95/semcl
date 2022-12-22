import argparse
from torchinfo import summary
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models

import swin_transformer

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(torchvision_models.__dict__[name]))
model_names = ['swin_tiny', 'swin_small', 'swin_base',
               'swin_large'] + torchvision_model_names


def load_moco_backbone(backbone: nn.Module, linear_keyword, args):
    # load state_dict
    checkpoint = torch.load(args.full_ckpt, map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint["state_dict" if "state_dict" in checkpoint else "model"]
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        # For ddp models, they are wrapped by 'module.'
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # Otherwise, the ckpt dict starts with 'base_encoder'
        elif k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = backbone.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"%s.weight" % linear_keyword,
                                     "%s.bias" % linear_keyword}, f"Missing keys: {msg.missing_keys}"
    # It's fine to ignore above missing keys warning
    # see https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/issues/124#issuecomment-992111842
    return backbone


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    par_dir = os.path.dirname(filename)
    if os.path.exists(par_dir) is not True:
        os.makedirs(par_dir)

    torch.save(state, filename)


def moco2bkb(args):
    r"""moco2bkb
    Use this function to extract base encoder backbone from pretrained weights
    """
    # Retrieve pretrained weights
    assert (len(args.full_ckpt) > 0), "You have to specify pretrained ckpt path."
    # check existence of full ckpt
    assert (os.path.isfile(args.full_ckpt)
            ), f"Given full checkpoint at {args.full_ckpt} does not exist."
    print(f"Extracting backbone from pretrained checkpoint {args.full_ckpt}.")

    # create model
    print(f"=> creating model '{args.arch}'")
    if args.arch.startswith('swin'):
        # Unlike moco whose pretrained weights contain both base and momentum encoder,
        # swin transformer pretrained weights contains only the backbone (base encoder) itself.
        model = swin_transformer.__dict__[args.arch]()
        linear_keyword = 'head'
        model = load_moco_backbone(
            model, linear_keyword=linear_keyword, args=args)
    else:
        print(
            f"output_stride is {args.output_stride if args.output_stride==8 or args.output_stride==16 else None}.")
        # from deeplabv3plus. See https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/4e1087de98bc49d55b9239ae92810ef7368660db/network/modeling.py#L34.
        if args.output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
        elif args.output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        else:  # default resnet. See https://github.com/pytorch/vision/blob/5b4f79d9ba8cbeeb8d6f0fbba3ba5757b718888b/torchvision/models/resnet.py#L186.
            replace_stride_with_dilation = None
        model = torchvision_models.__dict__[args.arch](
            zero_init_residual=True, replace_stride_with_dilation=replace_stride_with_dilation)
        linear_keyword = 'fc'
        model = load_moco_backbone(
            model, linear_keyword=linear_keyword, args=args)

    summary(model, input_size=(1, 3, 224, 224), device="cpu")
    if args.summary_only:
        print("In 'summary_only' mode, no checkpoint will be saved.")
    else:
        slash_idx = str(args.full_ckpt).rfind('/')
        bkb_filename = os.path.join(os.path.dirname(
            args.full_ckpt), "bkb_"+os.path.basename(args.full_ckpt))
        save_checkpoint(
            {'state_dict': model.state_dict(), }, filename=bkb_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument("--output-stride", type=int, default=-1, choices=[-1, 8, 16],
                        help="This option is valid for only resnet backbones. -1: no output stride (default resnets).")
    parser.add_argument('full_ckpt', default='', type=str, metavar='PATH',
                        help="Path to pretrained weights having same architecture with --arch option.")
    parser.add_argument('--summary-only', action='store_true',
                        help='Print backbone summary without saving.')
    args = parser.parse_args()

    moco2bkb(args)
