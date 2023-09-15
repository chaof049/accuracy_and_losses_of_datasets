import torch
import numpy as np
import torch.nn as nn #all neural network modules, nn.Linear, nn.Conv2d, Batch Norm, Loss functions
import torch.optim as optim #for all optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F #all functioins that don't have any parameters
from torch.utils.data import DataLoader #gives easier dataset management and creates mini batches
import torchvision
import torchvision.transforms as transforms #transformation we can perform on our dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from matplotlib import pyplot as plt

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
in_channel = 3
num_classes = 3
learning_rate = 1e-3
batch_size = 12
num_epochs = 8
#X = 0.8
mean = [0.3191, 0.3242, 0.4615]
std = [0.2011, 0.1530, 0.1456]

#create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels = in_channel, num_classes = num_classes):
        super(CNN, self).__init__()
        #first layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        #second layer
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #maxpool
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #drop layer
        self.drop = nn.Dropout2d(p=0.2)
        #flatten image
        self.fc = nn.Linear(56*56*24, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.dropout(self.drop(x), training=self.training)
        #flatten
        x = x.view(-1,56*56*24)
        x = self.fc(x)

        return torch.log_softmax(x, dim=1)


#dataset path
train_dataset_path = "/Users/supremepradhananga/Pycharm/custom/train"
test_dataset_path = "/Users/supremepradhananga/Pycharm/custom/test"

#transform image size
train_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                      ])

test_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                     ])

#load data
train_dataset = ImageFolder(root=train_dataset_path, transform=train_transform)
test_dataset = ImageFolder(root=test_dataset_path, transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = 8,
                                          shuffle = False,
                                          )

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = 8,
                                          shuffle = False,
                                          )

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

model = CNN().to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                      lr = learning_rate)

# train network

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)

        # get to correct shape
        # data = data.reshape(data.shape[0], -1), for 2d data only

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    # cost.append(sum(losses)/len(losses))
    print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')


# check accuracy

def check_accuracy(loader, model):
    if loader.dataset:
        print("checking accuracy on training data")
    else:
        print("chcecking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = "{:.2f}".format(float(num_correct) / float(num_samples) * 100)
    print(f'got {num_correct}/{num_samples} with accuracy {accuracy}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)