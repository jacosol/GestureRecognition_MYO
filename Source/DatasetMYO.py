import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
from scipy import signal
# our custom Dataset class
class DatasetMYO(Dataset):
    
    def __init__(self,datapath,transform,sample_len=2500):
        self.sample_len = sample_len
        self.path = datapath
        self.files = [y for x in os.walk(self.path) 
                        for y in glob.glob(os.path.join(x[0], '*.npy'))]
        self.len = len(self.files)
        self.transform = transform
        
    def __getitem__(self, index):
        sample = np.load(self.files[index])
        label = int(self.files[index].split('_')[-2][-1])
        if len(sample[0]) < 2500:
            #pad with zeros if the length is smaller than 2500
            sample = np.array([np.pad(channel,(0,self.sample_len - len(channel)),'constant') for channel in sample])

        sample = sample - np.mean(sample, axis=1).reshape(8,1)
        sample = sample / np.std(sample, axis=1).reshape(8,1)


        label = torch.LongTensor([label])[0]
            
        return(sample, label)
        
    
    def __len__(self):
        return self.len
    
#convert th einput to tnesor for torchvision handling
class ToTensor_(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.from_numpy(sample).float()