import DatasetMYO

class ClassifierMYO(nn.Module):
    
    def __init__(self):
        
        super(ClassifierMYO, self).__init__()
        self.conv1 = nn.Conv1d(8,16,55,padding = 24)
        self.conv2 = nn.Conv1d(16,32,25,padding = 12)
        self.pool = nn.MaxPool1d(2, 2)
        # linear layer (32 * 2500/2/2  -> 500)
        self.fc1 = nn.Linear(32*623, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        # flatten image input

        x = x.view(15,32*623)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.log_softmax(self.fc2(x), dim=1)
        return x