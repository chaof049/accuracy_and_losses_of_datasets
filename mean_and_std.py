import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

train_dataset_path = "/Users/supremepradhananga/Pycharm/custom/train"
train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
#load data
train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = 8,
                                          shuffle = False,
                                          )

def get_mean_and_std(loader):
    mean = 0
    std = 0
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        #print(images.shape)
        mean += images.float().mean(2).sum(0)
        std += images.float().std(2).sum(0)
        total_images_count += image_count_in_a_batch

        mean /= total_images_count
        std /= total_images_count

        return mean, std

mean, std = get_mean_and_std(train_loader)
print(mean)
print(std)