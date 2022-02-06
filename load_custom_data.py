import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from torchvision.io import read_image
import pandas as pd
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, list_filename, img_dir, transform=None, target_transform=None):
        self.img_lists = pd.read_csv(list_filename)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        anchor_img_path = os.path.join(self.img_dir, self.img_lists.iloc[idx, 0])
        nanchor_img_path = os.path.join(self.img_dir, self.img_lists.iloc[idx, 1])
        anchor_image = Image.open(anchor_img_path)
        nanchor_image = Image.open(nanchor_img_path)
        # label = self.img_lists.iloc[idx, 1]
        if self.transform:
            anchor_image_trans = self.transform(anchor_image)
            nanchor_image_trans = self.transform(nanchor_image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return anchor_image_trans, nanchor_image_trans