import torch
import numpy as np
import torch.nn as nn #all neural network modules, nn.Linear, nn.Conv2d, Batch Norm, Loss functions
import torch.optim as optim #for all optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F #all functioins that don't have any parameters
from torch.utils.data import DataLoader #gives easier dataset management and creates mini batches
import torchvision
import torchvision.transforms as transforms #transformation we can perform on our dataset
import torchvision.datasets as datasets
#from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 12
num_epochs = 15
#X = 0.8
mean = [0.3191, 0.3242, 0.4615]
std = [0.2011, 0.1530, 0.1456]

#create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels = in_channel, num_classes = num_classes):
        super(CNN, self).__init__()
        #first layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        #second layer
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #third layer
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #maxpool
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #drop layer
        self.drop = nn.Dropout2d(p=0.2)
        #flatten image
        self.fc = nn.Linear(3*3*36, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.dropout(self.drop(x), training=self.training)
        #flatten
        x = x.view(-1,3*3*36)
        x = self.fc(x)

        return torch.log_softmax(x, dim=1)

#loadData
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = CNN().to(device)

#loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr = learning_rate,
                      momentum = 0.9)

# train network

train_accuracy = []
train_losses = []


def train(epoch):
    print('\n Epoch : %d' % epoch)

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for data in tqdm(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    train_accuracy.append(accuracy)
    train_losses.append(train_loss)
    print('Train Loss : %.3f | Accuracy : %.3f' % (train_loss, accuracy))


# test network

eval_losses = []
eval_accuracy = []


def test(epoch):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    eval_losses.append(test_loss)
    eval_accuracy.append(accuracy)

    print('Test Loss : %.3f | Accuracy : %.3f' % (test_loss, accuracy))


epochs = num_epochs
for epoch in range (1, epochs+1):
    train(epoch)
    test(epoch)

#accuracy_plot
plt.plot(train_accuracy, '-o')
plt.plot(eval_accuracy, '-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train', 'Valid'])
plt.title('Train vs Valid Accuracy')
plt.show()

#losses_plot
plt.plot(train_losses, '-o')
plt.plot(eval_losses, '-o')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train', 'Valid'])
plt.title('Train vs Valid Losses')
plt.show()
