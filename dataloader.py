import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import cv2
from PIL import Image
import helper

data_dir = 'MarioMapImg'

# train_transforms = transforms.Compose([
#                                 transforms.RandomRotation(30),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.ToTensor()])
# test_transforms = transforms.Compose([transforms.Resize(224, 3056),
#                                       transforms.ToTensor()])

# train_data = datasets.ImageFolder(data_dir + '/level1',
#                                     transform=train_transforms)
# test_data = datasets.ImageFolder(data_dir + '/level2',
#                                     transform=test_transforms)

# #Data Loading
# trainloader = torch.utils.data.DataLoader(train_data,
#                                                    batch_size=32)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# data_iter=iter(testloader)
# images, labels = next(data_iter)

transform = transforms.Compose([transforms.Resize((224, 3056)),
transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)