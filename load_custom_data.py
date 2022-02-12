import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from torchvision.io import read_image
import pandas as pd
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, datasets=['coco', 'ade']):
        self.img_lists = pd.DataFrame(columns = ['anchor','nanchor'])
        for OneDataset in datasets:
            if OneDataset=='voc':
                VOCImgListCSV=pd.read_csv(os.path.join(img_dir,"VOC_ImgList.csv"))
                self.img_lists=self.img_lists.append(VOCImgListCSV,ignore_index=True)
                print("VOC2012 added")
            elif OneDataset=='coco':
                COCOImgListCSV=pd.read_csv(os.path.join(img_dir,"COCO_ImgList.csv"))
                self.img_lists=self.img_lists.append(COCOImgListCSV,ignore_index=True)
                print("COCO added")
            elif OneDataset=='ade':
                ADEImgListCSV=pd.read_csv(os.path.join(img_dir,"ADE_ImgList.csv"))
                self.img_lists=self.img_lists.append(ADEImgListCSV,ignore_index=True)
                print("ADE20K added")
            else:
                raise ValueError("Unrecognize dataset choice.")
        
        print("%d samples will be used." % len(self.img_lists))

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