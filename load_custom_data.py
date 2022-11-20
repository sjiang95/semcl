from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile
# https://github.com/python-pillow/Pillow/issues/1510#issuecomment-151458026
# Add the following line to fix a pillow error
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, datasets=['coco', 'ade', 'voc']):
        self.img_dir = img_dir
        self.paired_samples=[]
        for OneDataset in datasets:
            if OneDataset=='voc':
                sample_list_file=os.path.join(self.img_dir,"VOC_ImgList.txt")
                voc_sample_list=open(sample_list_file,'r')
                count=0
                for oneLine in voc_sample_list.readlines():
                    oneLine=oneLine.strip()
                    self.paired_samples.append(oneLine)
                    count+=1
                print(f"{count} VOC2012 pairs added.")
            elif OneDataset=='coco':
                sample_list_file=os.path.join(self.img_dir,"COCO_ImgList.txt")
                COCO_sample_list=open(sample_list_file,'r')
                count=0
                for oneLine in COCO_sample_list.readlines():
                    oneLine=oneLine.strip()
                    self.paired_samples.append(oneLine)
                    count+=1
                print(f"{count} COCO pairs added.")
            elif OneDataset=='ade':
                sample_list_file=os.path.join(self.img_dir,"ADE_ImgList.txt")
                ADE_sample_list=open(sample_list_file,'r')
                count=0
                for oneLine in ADE_sample_list.readlines():
                    oneLine=oneLine.strip()
                    self.paired_samples.append(oneLine)
                    count+=1
                print(f"{count} ADE20K pairs added.")
            elif OneDataset=='city':
                sample_list_file=os.path.join(self.img_dir,"Cityscapes_ImgList.txt")
                City_sample_list=open(sample_list_file,'r')
                count=0
                for oneLine in City_sample_list.readlines():
                    oneLine=oneLine.strip()
                    self.paired_samples.append(oneLine)
                    count+=1
                print(f"{count} Cityscapes pairs added.")
            else:
                raise ValueError("Unrecognize dataset choice.")
        
        print("%d samples will be used." % len(self.paired_samples))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        rel_anchor_img_path, rel_nanchor_img_path = self.paired_samples[idx].split(',')
        anchor_img_path = os.path.join(self.img_dir, rel_anchor_img_path)
        nanchor_img_path = os.path.join(self.img_dir, rel_nanchor_img_path)
        anchor_image = Image.open(anchor_img_path)
        nanchor_image = Image.open(nanchor_img_path)
        if self.transform:
            anchor_image_trans, nanchor_image_trans = self.transform(anchor_image, nanchor_image)
        return anchor_image_trans, nanchor_image_trans