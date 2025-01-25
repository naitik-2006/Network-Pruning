# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:25:37 2024

@author: jishu
"""


import os

from PIL import Image
from torch.utils.data import Dataset , DataLoader


class TinyImageNet(Dataset):
    def __init__(self , root_dir , transform = None , split = 'train'):
        
        super(TinyImageNet , self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = os.listdir(os.path.join(root_dir , split))
        self.classes_to_idx = {cls : i for i , cls in enumerate(self.classes)}
        self.images = self._load_images()
        
    def _load_images(self):
        images = []
        
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir , self.split , cls , 'images')
            for image_file in os.listdir(cls_dir):
                image_path = os.path.join(cls_dir , image_file)
                images.append((image_path , self.classes_to_idx[cls]))
                
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self , idx):
        image_path , label = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        if(self.transform):
            image = self.transform(image)
        
        return image , label
    
def get_loader(
    
    root_folder,
    transform,
    batch_size=32,
    num_workers=12,
    
    shuffle=True,
    pin_memory=True,
    train = True
):

    dataset = TinyImageNet(root_folder,transform=transform)
    
    if(train == False):
        batch_size = 1
    
    loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    return loader,dataset
    
    
        
        