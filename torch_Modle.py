import torch 
# from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os



    # Get data 
    video = cv2.VideoCapture("attack_highdef_client001_session01_highdef_photo_adverse.mov")

    net = Net()

    # Initialize a list to store the frames
    frames = []

    # Loop over the frames in the video
    for i in range(100):
        # Read the next frame
        success, frame = video.read()
        if not success:
            break
        
        # Preprocess the frame
        # frame = cv2.resize(frame, (224, 224))  # Resize the frame
        frame = frame.transpose((2, 0, 1))  # Transpose the frame to match the input dimensions of the network
        frame = torch.from_numpy(frame).float()  # Convert the frame to a tensor
        frame = frame.unsqueeze(0)  # Add a batch dimension to the frame
        print(frame.shape)
        # Add the frame to the list of frames
        frames.append(frame)([1, 3, 240, 320])

    # Concatenate the frames into a single tensor
    frames = torch.cat(frames, dim=0)

    print(frames.shape) #([100, 3, 240, 320])
    #1,28,28 - classes 0-9











# Image Classifier Neural Network
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
    

