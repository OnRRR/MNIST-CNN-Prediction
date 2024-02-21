import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):

        #super(Net, self).__init__()
        super().__init__()
        #Convolution Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3))
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(3,3))
        self.conv6 = nn.Conv2d(32, 32, kernel_size=(3,3))
        self.conv7 = nn.Conv2d(32,32,kernel_size=(3,3))

        #1D and 2D Dropout Operations
        #self.conv2_drop = nn.Dropout2d()
        self.fcDout = nn.Dropout(p=0.2) # fc -> fully connected

        #Max Pooling
        self.pool2d = nn.MaxPool2d(2,stride=2)
        self.bn2 = nn.BatchNorm2d(20) # Its 20 because we'll use it after self.conv2
        #self.bn4 = nn.BatchNorm3d(60) # Its 60 because we'll use it after self.conv4

        #Linear Layers
        self.fc1 = nn.Linear(1152, 256)  # w1 -> (256,1152)
        self.fc2 = nn.Linear(256,128)  # w2 -> (128,256)
        self.fc3 = nn.Linear(128,64)  # w3 -> (64,128)
        self.fc4 = nn.Linear(64,6)  # w4 -> (6,64)

    def forward(self, x):
        #Calculate first convolution parameters
        x = self.conv1(x)
        x = F.relu(x)
        #x = self.pool2d(x)
        #x = F.relu(x)

        #Calculate second convolution parameters
        x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.conv2_drop(x) # We used dropout here to avoid overfitting
        x = F.relu(x)
        x = self.pool2d(x)
        #x = F.relu(x)

        #Calculate third convolution parameters
        x = self.conv3(x)
        x = F.relu(x)
        #x = self.pool2d(x)
        #x = F.relu(x)

        #Calculate fourth convolution parameters
        x = self.conv4(x)
        #x = self.conv2_drop(x)
        #x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2d(x)

        #Calculate fifth convolution parameters
        x = self.conv5(x)
        x = F.relu(x)
        #x = self.pool2d(x)
        #x = F.relu(x)
        #print(x.size())

        #Calculate sixth convolution parameters
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool2d(x)

        #Calculate seventh convolution parameters
        x = self.conv7(x)
        x = F.relu(x)
        x = self.pool2d(x)
        #print(x.size())

        #Vectorize the second convolution parameters
        x = x.view(-1,1152) # We didn't use x.copy() here because it would overflow the memory

        #Calculate fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fcDout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fcDout(x)
        x = self.fc3(x)
        x = self.fcDout(x)
        x = F.relu(x)
        x = self.fc4(x)

        #x = self.fcDout(x)
        #x = self.fc4(x)

        return torch.log_softmax(x, dim=1)