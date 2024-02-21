import torchvision
from torch.utils.data import DataLoader
from CNNModel import *
import torch.optim as opt
import torch.nn.functional as F
#import os (You can make file operations using this library)


if __name__ == '__main__':

    #Training Parameters
    num_epochs = 3
    batch_sizeTrain = 64
    batch_sizeTest = 1000 # If you don't have Memory (RAM) limitations you  don't have to use this line
    #batch_sizeTest = all
    learning_rate = 0.01
    momentum = 0.9 #Default momentum
    #print_freq = 50 (Use this if you want to analyze training process)
    #device = "cuda"
    device = "cpu"  #(Use this is if you don't want to use gpu memory for training)

    #Downloading Data and Creating DataLoader
    trainLoader = DataLoader(torchvision.datasets.MNIST("../nnPytorch/data/",train=True, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.Grayscale(),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0,), (1,))
                                                        ])),
                             batch_size=batch_sizeTrain, shuffle=True)

    testLoader = DataLoader(torchvision.datasets.MNIST("../nnPytorch/data/",train=False, download=True,
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.Grayscale(),
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((0,), (1,))
                                                       ])),
                            batch_size=batch_sizeTest, shuffle=True)

    # Training and Testing
    model = Net().to(device=device)

    optimizer = opt.SGD(params=model.parameters(),lr=learning_rate, momentum=momentum) # Stochastic gradient descent optimizer

    # Training The Model
    model.train() # Train Mode
    for epoch in range(num_epochs):
        for i,(batch,target) in enumerate(trainLoader): # Target = Label
            print(batch.size())
            optimizer.zero_grad() # Zero the Gradients (for each batch)
            output = model.forward(batch.to(device))
            loss = F.nll_loss(output, target.to(device)) # nll_loss = negative_cross_entropy_loss
            loss.backward() #Backpropagation respectively to optimizer
            optimizer.step() # Update the Parameters

    # Testing
    model.eval() # Evaluate the model
    correct = 0 # Number of correct predictions
    total = 0
    test_loss = 0 # Value of the test loss

    for i,(batch,target) in enumerate(testLoader): # Target = Label
        output = model.forward(batch.to(device))
        #test_loss += F.nll_loss(output, target.to(device)) # nll_loss = negative_cross_entropy_loss
        #total += target.size(0)
        pred = output.data.max(1, keepdim=True)[1] # Taking max value of outputs for each data and getting its location
        correct += pred.eq(target.to(device).data.view_as(pred)).sum() # Comparing predicted values with target values

    test_loss /= len(testLoader.dataset)

    print(f"\n Test Set: Accuracy: {100. * correct/len(testLoader.dataset):.3f}")






