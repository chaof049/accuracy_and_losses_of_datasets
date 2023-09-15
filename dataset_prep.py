import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

train_dataset_path = "/Users/supremepradhananga/Pycharm/custom/train"

train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor()
                                       ])

train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform = train_transforms)

def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size = 12, shuffle = True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid.numpy(),(1,2,0)))
    print('labels: ', labels)

show_transformed_images(train_dataset)