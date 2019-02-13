from ClassifierMYO import *
from DatasetMYO import *
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.autograd import Variable

"""
Attribute Information:

Description of raw_data _*** file 
Each file consist of 10 columns: 
1) Time - time in ms; 
2-9) Channel - eightEMG channels of MYO Thalmic bracelet; 
10) Class for the label of gestures: 
0 - unmarked data, 
1 - hand at rest, 
2 - hand clenched in a fist, 
3 - wrist flexion, 
4 - wrist extension, 
5 - radial deviations, 
6 - ulnar deviations, 
7 - extended palm (the gesture was not performed by all subjects).
"""

new_train = 1

validataionDS_size = 0.05
batch_size = 20
epochs = 4

processed_ds_path = '../CustomDataset/processed'

DSMYO = DatasetMYO(processed_ds_path, None)
DSMYO_test = DatasetMYO(processed_ds_path, None)
num_train = len(DSMYO)
indices = list(range(num_train))
indices = indices[:len(indices) - len(indices)%batch_size]
np.random.shuffle(indices)
split = int(np.floor(validataionDS_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(DSMYO, batch_size=batch_size,
    sampler=train_sampler, drop_last=True)
valid_loader = DataLoader(DSMYO, batch_size=batch_size, 
    sampler=valid_sampler, drop_last=True)
test_loader = DataLoader(DSMYO_test, batch_size=batch_size, shuffle = True, drop_last=True)

if new_train:
    model_with_val = ClassifierMYO()
    model_with_val.double()
    model_with_val.to('cuda')

# train script with validation 

criterion = nn.NLLLoss()
optimizer = optim.Adam(model_with_val.parameters())


train_loss_overtime = []
valid_loss_overtime = []
for e in range(epochs):
    
    train_loss = 0.0
    valid_loss = 0.0
    model_with_val.train()
    status = 0
    for sample, label in train_loader:
        status += 1
        if status%10 == 0:
            print(f'epoch progress: {int (status*batch_size/len(train_loader.dataset)*100)}%')
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        sample = Variable(sample.to('cuda'))
        output = model_with_val(sample)
        output.to('cuda')
        # calculate the batch loss
        label = Variable(label.to('cuda'))
        loss = criterion(output, label)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*sample.size(0)

    ######################    
    # validate the model #
    ######################
    model_with_val.eval()
    status = 0
    for data, target in valid_loader:

        status += 1
        if status%2 == 0:
            print(f'validating: {int (status*batch_size/len(valid_loader.sampler)*100)}%')
        # forward pass: compute predicted outputs by passing inputs to the model
        data = Variable(data.to('cuda'))
        output = model_with_val(data)
        output.to('cuda')
        target = Variable(target.to('cuda'))
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item()*data.size(0)


    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    valid_loss_overtime.append(valid_loss)
    train_loss_overtime.append(train_loss)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        e, train_loss, valid_loss))