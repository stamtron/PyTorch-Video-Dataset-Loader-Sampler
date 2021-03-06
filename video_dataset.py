import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import glob
import torch
from PIL import Image
import re
import torchvision

 
class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):
        indices = []
        for i in range(len(end_idx) - 1):
            start = end_idx[i]
            end = end_idx[i + 1] - seq_length
            if start > end:
                pass
            else:
                indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices
        
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.indices)
    
    
class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, temp_transform, spat_transform, tensor_transform, length, lstm=False, oned = False, augment = False, multi = 1): #csv_file, 
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.temp_transform = temp_transform
        self.spat_transform = spat_transform
        self.tensor_transform = tensor_transform
        self.length = length
        self.lstm = lstm
        self.oned = oned
        self.augment = augment
        self.multi = multi
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        #print('Getting images from {} to {}'.format(start, end))
        indices = list(range(start, end))
        images = []
        #tr = self.transform
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            images.append(image)
        x = images
        if not self.oned:
            x = self.temp_transform(x)
        if self.augment:
            x = self.spat_transform(x)
        x = self.tensor_transform(x)
        y = torch.tensor([self.image_paths[start][self.multi]], dtype=torch.long)
        y = y.squeeze(dim=0)
        y = y.float()
        #print(y.shape)
        if self.lstm:
            x = x.permute(1,0,2,3)
        if self.oned:
            x = x.squeeze(dim=1)
        return x, y
    
    def __len__(self):
        return self.length
