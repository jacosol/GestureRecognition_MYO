import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import os
import pandas as pd
import librosa as lb
from librosa import display
import scipy
from torch import nn
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

def validate_model(model, validation_loader):

	model.eval()
	model.to('cpu')
	accuracy = 0
	for sample, label in validation_loader:

		output = model(sample)
		topp, topc = torch.exp(output).topk(1, dim=1)

		print(label.numpy)
		print(np.hstack(topc))

        equals = topc == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f'accuracy : {int(accuracy/len(validation_loader)*100)}%')