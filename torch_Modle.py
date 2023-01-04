import torch 
import Data_set
from torch import nn, save, load
from torch.optim import Adam
import os
import numpy as np




class PAD(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)
    
    

data=[]
labels=[]

real_path = r"D:\db\training_tesnsors\real_tensors"
real_list = os.listdir(real_path)
for real in real_list:
    tensor = torch.load(real)
    data.append(tensor)
    labels.append(0)
    
hand_path = r"D:\db\training_tesnsors\hand_tensors"
hand_list = os.listdir(hand_path)
for hand in hand_list:
    tensor = torch.load(hand)
    data.append(tensor)
    labels.append(1)
    
    
fixed_path = r"D:\db\training_tesnsors\fixed_tensors"
fixed_list = os.listdir(fixed_path)
for fixed in fixed_list:
    tensor = torch.load(fixed)
    data.append(tensor)
    labels.append(2)

labels_tensor = torch.from_numpy(np.array(labels)).long()

dataset = Data_set(data, labels_tensor)

# Create a data loader to iterate over the dataset
dataloader = data.DataLoader(dataset, batch_size=200, shuffle=True)
# Instance of the neural network, loss, optimizer 
clf = PAD().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    for epoch in range(10): # train for 10 epochs
        for batch in dataset: 
            X,y = batch 
            X, y = X.to('cpu'), y.to('cpu') 
            yhat = clf(X) 
            loss = loss_fn(yhat, y) 

            # Apply backprop 
            opt.zero_grad()
            loss.backward() 
            opt.step() 

        print(f"Epoch:{epoch} loss is {loss.item()}")
    

