import argparse
import os
import sys
import requests
from io import BytesIO

from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
import swin_transformer


from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

pretrained_weight_url = {
    'swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
    'swin_small': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth',
    'swin_base': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
    'swin_large': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
}
model_names = pretrained_weight_url.keys()

def preprocess_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), cropSize=224) -> torch.Tensor:
    preprocessing = transforms.Compose([
        transforms.Resize(cropSize),
        transforms.CenterCrop(cropSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='swin_tiny',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: swin_tiny)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--pretrained', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument(
        '--img-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument("--crop-size", default=224, type=int, help="Resize image.")
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--output-dir', default='.', help='Path where to save visualizations.')
    parser.add_argument(
        '--method',
        type=str,
        default='scorecam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    
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

    model = swin_transformer.__dict__[args.arch](state_dict=checkpoint["state_dict" if "state_dict" in checkpoint else "model"])
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)

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
    input_tensor = preprocess_image(img, cropSize=args.crop_size)

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    # save attentions heatmaps
    args.output_dir=os.path.join(args.output_dir,"visualize_attention")
    os.makedirs(args.output_dir, exist_ok=True)
    fname = os.path.join(args.output_dir, str(os.path.basename(args.img_path)).split(sep='.')[0]+("_IN" if len(args.pretrained)==0 else "_SemCL")+f"_{args.method}" + ".png")
    cropRawImg=transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.CenterCrop(args.crop_size),
    ])
    croppedImg=cv2.cvtColor(np.array(cropRawImg(img)), cv2.COLOR_RGB2BGR)
    print(type(croppedImg))
    img2tensor=transforms.ToTensor()
    cam_image = show_cam_on_image(torch.permute(img2tensor(croppedImg),dims=(1,2,0)).numpy(), grayscale_cam)
    cv2.imwrite(fname, cam_image)
    print(f"Ploted img save to '{fname}'.")
    
    # save cropped raw img
    croppedImgFname=os.path.join(args.output_dir, str(os.path.basename(args.img_path)).split(sep='.')[0]+ "_224.png")
    if not os.path.exists(croppedImgFname):
        tensor2img=transforms.ToPILImage()
        print(f"Save cropped raw img to '{croppedImgFname}'.")
        cv2.imwrite(croppedImgFname,croppedImg)
