import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset 
from torchvision.transforms import Resize, Compose, ToTensor

class AnimeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fnames = os.listdir(self.root_dir)
        self.transform = Compose([
            ToTensor(),
            Resize(256)
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_name = self.fnames[index]
        img_path = os.path.join(self.root_dir, img_name)

        img = np.array(Image.open(img_path))
        input_img = self.transform(img[:,int(img.shape[-2]/2):,:])
        target_img = self.transform(img[:,:int(img.shape[-2]/2),:])
        
        return input_img, target_img