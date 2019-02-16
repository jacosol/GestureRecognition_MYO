import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import os
import pandas as pd
import scipy
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

def load_sample(path):
    """load a specific sample from the raw dataset
    
    Arguments:
    
    path : string, path to the sample in the raw dataset
    
    """
    
    data = pd.read_csv(path, sep='\t')
    data.columns = ['time', 'channel_1','channel_2','channel_3',
                    'channel_4','channel_5','channel_6','channel_7', 'channel_8', 'labels']
    
    return data

def extend_labeled_region(sample, insertL = 500):
    """Extend a labeled region of the EGM data
    
    Arguments:
    
    sample    : pandas dataframe representating a sample loaded with load_sample() or load_random_sample()
    insertL : int, magnitude of the extension on each side of the region
    """
    labels = set(sample['labels'])
    
    for j in list(labels)[1:]:
        try:
            inds = np.hstack(np.where(sample['labels'] == j)[0])
            inds = np.insert(inds,0,np.arange(inds[0]-insertL, inds[0]))
            jumps = np.hstack(np.where(np.diff(inds) > 2)[0])
            for jump in jumps:
                refLow = inds[jump-1]
                refHigh = inds[jump+1]
                #insert at the end of the first region
                inds = np.insert(inds, jump-1, np.arange(inds[jump-1], inds[jump-1] + insertL))
                #insert at the beginning of the second region
                inds = np.insert(inds, jump+1 + insertL, np.arange(inds[jump+1+insertL] - insertL, inds[jump+1+insertL]))
                #insert at the end of the second region
                inds = np.insert(inds, len(inds), np.arange(inds[-1] , inds[-1] + insertL))  
            sample['labels'].values[inds] = j
        except:
            print(f'no label = {j}')

    return sample


def create_single_txt_sample(file,index, save_root):
    """Loads raw data and splits it into labeled regions creating a sample
    
    Arguments:
    
    file    : string, path to the raw data txt
    index   : int, number of the raw data file used for naming
    """    
    
    sample = load_sample(file)
    sample = extend_labeled_region(sample)
    
    for label in list(set(sample['labels']))[1:]:
        try:
            inds = np.hstack(np.where(sample['labels'] == label)[0])
        except:
            print('bad sample')
            return False
            
            
        jumps = np.hstack(np.where(np.diff(inds) > 2)[0])
        for channel in range(1,9):
            key = 'channel_' + str(channel)
            #isolate first repeat of gesture
            data1 = sample[key][inds[:jumps[0]]]
            #isolate second repeat of gesture
            data2 = sample[key][inds[jumps[0]+1:len(inds)]]
            #clip the lenght at 2500 and if shorter pad with zeros
            try:
                data1 = data1[:2500]
            except:
                data1 = np.pad(data1,(0,2500-len(data1)))
                print('shortdata')
            try:
                data2 = data2[:2500]
            except:
                data2 = np.pad(data2,(0,2500-len(data2)))
                print('shortdata')
            #save each repeat in the labeled folder in the database
            np.savetxt(os.path.join(save_root,str(int(label)),''.join([key,'_',str(index).zfill(3),'.txt'])), data1)
            np.savetxt(os.path.join(save_root,str(int(label)),''.join([key,'_',str(index+1).zfill(3),'.txt'])), data2)
    return True


def create_dataset_dirs(path):
    """create empty dirs corresponding to the labels

    Arguments: 
        path : string, path where to create dirs corresponding to labels

    """
    for i in range(1,8):
        os.mkdir(os.path.join(path, str(i)))

def process_data_set(raw_ds_path, processed_ds_path):
    """stacks each 8 channel and label for each smaple into a tuple. the tuple has a (8,2500) data array and (1) label 

    Arguments:
        raw_ds_path : 
        processed_ds_path :

    """
    try:
        create_dataset_dirs(processed_ds_path)
    except:
        print('WARNING: directories corresponding to labels 1-7 already exist')
    
    folders = np.sort(glob.glob(os.path.join(raw_ds_path,'*')))
    labels = np.sort([x for x in os.listdir(raw_ds_path) if len(x)<2])
    savefolders = np.sort((glob.glob(os.path.join(processed_ds_path,'*'))))
    
     
    for folder, savefolder, label in zip(folders, savefolders, labels):
        files = glob.glob(os.path.join(folder,'*'))
        Nfiles = len(files)/8
        if Nfiles%int(Nfiles) != 0:
            print('WARNING: dat least one sample has not exactly 8 channels')
        for i in range(1,int(Nfiles)):
            index = str(i).zfill(3)
            batch = glob.glob(os.path.join(folder,'*' + str(index).zfill(3) +'.txt'))
            datalist = [np.loadtxt(file) for file in batch]
            data = np.array(datalist)
            np.save(os.path.join(savefolder, label + '_' + index + '.npy'), data)
        print(f'saving formatted data in {savefolder}')