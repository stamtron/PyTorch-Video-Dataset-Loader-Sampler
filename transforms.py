import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import glob
import torch
from PIL import Image
import re
import matplotlib.pyplot as plt
import torchvision

import sys
sys.path.append('../../torch_videovision/')

from torchvideotransforms.video_transforms import Compose as vidCompose
from torchvideotransforms.video_transforms import Normalize as vidNormalize
from torchvideotransforms.video_transforms import Resize as vidResize
from torchvideotransforms.volume_transforms import ClipToTensor

import vidaug.augmentors as va

from video_dataset import *


def get_tensor_transform(finetuned_dataset, resize = False):
    if finetuned_dataset == 'ImageNet':
        video_transform_list = [
            ClipToTensor(channel_nb=3),
            vidNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if resize:
            video_transform_list.insert(0,vidResize((288,352)))
    if finetuned_dataset == 'Kinetics':
        norm_value=255
        video_transform_list = [
            ClipToTensor(channel_nb=3),
            vidNormalize(mean=[110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value], std=[38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value]),
        ]
        if resize:
            video_transform_list.insert(0,vidResize((288,352)))
    tensor_transform = vidCompose(video_transform_list)
    return tensor_transform


def get_temporal_transform():
    temp_transform = va.OneOf([
        va.TemporalBeginCrop(size=16),
        va.TemporalCenterCrop(size=16),
        va.TemporalRandomCrop(size=16),
        va.TemporalFit(size=16),
        va.Sequential([
            va.TemporalElasticTransformation(),
            va.TemporalFit(size=16),
        ]),
        va.Sequential([     
            va.InverseOrder(),
            va.TemporalFit(size=16),
        ]),
    ])
    return temp_transform


def get_spatial_transform(n):
    transform = va.SomeOf([
        va.RandomRotate(degrees=20), #andomly rotates the video with a degree randomly choosen from [-10, 10]  
        va.HorizontalFlip(),# horizontally flip the video with 100% probability
        va.ElasticTransformation(0.1,0.1),
        va.GaussianBlur(sigma=0.1),
        #va.InvertColor(),
        #va.Superpixel(0.2,2),
        va.OneOf([
            va.Multiply(1.5),
            va.Multiply(0.75),
        ]),
        va.Add(10),
        va.Pepper(),
        va.PiecewiseAffineTransform(0.3,0.3,0.3),
        va.Salt(),
    ], N=n)
    return transform
