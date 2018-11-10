## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Convolution Layers =>  Output size = (W-F)/S +1
        ## Image W = 128
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 2)
        
        # Maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully connected linear layer
        self.fc1 = nn.Linear(18432, 1024) # Input taken from => print ("Flatten Layer:", x.shape)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)
        
        # Dropout layers
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
         
        # conv1 > relu > pool > drop1 
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        # conv2 > relu > pool > drop2
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        # conv3 > relu > pool > drop3
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        # conv4 > relu > pool > drop4
        x = self.drop4(self.pool(F.relu(self.conv4(x))))

        # Flatten for linear layer
        x = x.view(x.size(0), -1)
        # print ("Flatten Layer:", x.shape)
        
        # Dense > relu > drop5
        x = self.drop5(F.relu(self.fc1(x)))
        # Dense > relu > drop6
        x = self.drop6(F.relu(self.fc2(x)))
        # FC Layer
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
