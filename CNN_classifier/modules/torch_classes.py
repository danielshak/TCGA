import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class TumorDataset(Dataset):
    #reads in labels from dataframe and samples from csv 1 row at a time
    
    def __init__(self, labels_path, data_path, data_dims, genes_path, transform=None):
        labels_df = pd.read_pickle(labels_path)
        labels = np.array(labels_df).astype(int).T #one hot
        self.labels = labels
        self.int_labels = np.argmax(labels,axis=1) #converts from one hot to integer
        self.data_memmap = np.memmap(data_path, dtype='float64', mode='r', shape=data_dims)
        self.genes = pd.read_csv(genes_path, nrows=1, index_col=0).columns.values
        self.transform = transform
    
    def __len__(self):
        return (self.labels).shape[0]
    
    def __getitem__(self, idx):
        #Single row read from the memmap
        sample = self.data_memmap[idx,:]
        #sample = pd.read_csv(self.train_path, nrows=1, index_col=0, skiprows=range(0,idx)).values
        label = self.labels[idx,:]
        train_sample = {'data': sample, 'label': label}
        
        if self.transform:
            train_sample = self.transform(train_sample)
                        
        return train_sample

    def get_data(self, idx):
        #Single row read from the memmap
        sample = self.data_memmap[idx,:]
        #sample = pd.read_csv(self.train_path, nrows=1, index_col=0, skiprows=range(0,idx)).values
        label = self.labels[idx,:]
        train_sample = {'data': sample, 'label': label}
        
        if self.transform:
            train_sample = self.transform(train_sample)
                        
        return train_sample



class ToImage(object):
    
    """
    Maybe reordering here by gene position
    
    Follows the same image scaling as people in the paper.
    features_padded = features_train[i, :]/max(features_train[i, :])
    img = features_padded.reshape(102, 102)
    (img - 0.5)/0.5
    
    -Update
    -I switched this to having data already normalized (x-x.mean)/x.std
    -Only need reshape here
    """
    
    def __init__(self,annotation_file=None):
        self.file = annotation_file
    
    def __call__(self,sample):
        image, label = sample['data'], sample['label']
        image = np.reshape(image, (102,102))
        
        #add sorting later with annotation_file
        
        return {'data': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, label = sample['data'], sample['label']
        return {'data': torch.tensor(image), 'label': torch.tensor(label)}


class Net(nn.Module):

    def __init__(self, num_of_classes):
        super(Net, self).__init__()
        # input image channel, output channels, kernel size square convolution
        # kernel
        # input size = 102, output size = 100
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # input size = 50, output size = 48
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # input size = 24, output size = 24
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.vp = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256*12*12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_of_classes)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.bn1(self.vp(self.conv1(x))))
        x = F.relu(self.bn2(self.vp(self.conv2(x))))
        x = F.relu(self.bn3(self.vp(self.conv3(x))))
        x = self.drop2D(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
Net(num_of_classes=33)