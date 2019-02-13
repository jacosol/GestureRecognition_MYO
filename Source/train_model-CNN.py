import ClassifierMYO
import DatasetMYO

"""
Attribute Information:

Description of raw_data _*** file 
Each file consist of 10 columns: 
1) Time - time in ms; 
2-9) Channel - eightEMG channels of MYO Thalmic bracelet; 
10) Class â€“thelabel of gestures: 
0 - unmarked data, 
1 - hand at rest, 
2 - hand clenched in a fist, 
3 - wrist flexion, 
4 - wrist extension, 
5 - radial deviations, 
6 - ulnar deviations, 
7 - extended palm (the gesture was not performed by all subjects).
"""


validataionDS_size = 0.15
batch_size = 15

processed_ds_path = '../../DataGestureRecognition/CustomDataset/processed'

DSMYO = DatasetMYO(processed_ds_path, transform)
DSMYO_test = DatasetMYO(processed_ds_path, transform)
num_train = len(DSMYO)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(validataionDS_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(DSMYO, batch_size=batch_size,
    sampler=train_sampler)
valid_loader = DataLoader(DSMYO, batch_size=batch_size, 
    sampler=valid_sampler)
test_loader = DataLoader(DSMYO_test, batch_size=batch_size, shuffle = True)

model_with_val = ClassifierMYO()
model_with_val.double()

# train script with validation 

criterion = nn.NLLLoss()
optimizer = optim.Adam(model_with_val.parameters(), lr=0.01)

epochs = 30
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
            print(f'epoch progress: {int (status*15/len(dl.dataset)*100)}%')
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        try:
            output = model_with_val(sample)
            # calculate the batch loss
            loss = criterion(output, label)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*sample.size(0)
        except:
            print(f'error when computing the output. sample.shape = {sample.shape}')
    ######################    
    # validate the model #
    ######################
    model_with_val.eval()
    status = 0
    for data, target in valid_loader:
        try:
            status += 1
            if status%2 == 0:
                print(f'validating: {int (status*15/len(valid_loader.sampler)*100)}%')
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model_with_val(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        except:
            print(f'error when computing the output in validation. sample.shape = {sample.shape}')
        
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    
    valid_loss_overtime.append(valid_loss)
    train_loss_overtime.append(train_loss)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        e, train_loss, valid_loss))