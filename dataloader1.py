import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from utils import * 
from model import * 
from PIL import Image


# Options
base_path = './VGLC/Processed'
tile_width = 10
tile_height = 10
verbose = False

class MyDataset(Dataset):
    def __init__(self, datapath, train=False, transform=None):
        # self.train = train
        # if self.train:
        #     data_file = self.training_file
        # else:
        #     data_file = self.test_file
        self.data = []
        for file_name in os.listdir(datapath): 
            with open(base_path + "/" + file_name, 'r') as f:
               res = np.array( list( map( lambda l: [ord(c) for c in l.strip()], f.readlines() ) ) )
            self.data.append(res)
        self.transform = transform
            
        #load data and target .. (self.x) + loop
        
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img, mode="L")
        if self.transform is not None:
            img = self.transform(img)
        # img = torch.from_numpy(img)
        return img, None
    
    def __len__(self):
        return len(self.data)

m_transforms = transforms.Compose([
    transforms.RandomCrop(size=(14, 30)),
    transforms.ToTensor()
])
txtmap_dataset = MyDataset(base_path, transform=m_transforms)
print(txtmap_dataset[2])