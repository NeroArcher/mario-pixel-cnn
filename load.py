import torch
import torch.nn as nn
from model import *
from utils import * 
from PIL import Image
import torchvision.transforms.functional as TF

FILE = "models/pcnn_lr:0.00020_nr-resnet3_nr-filters80_0.pth"

obs = (1, 28, 28)
loaded_model = PixelCNN(nr_resnet=3, nr_filters=80, 
            input_channels=obs[0], nr_logistic_mix=10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  loaded_model = torch.nn.DataParallel(loaded_model)
  load_part_of_model

loaded_model.to(device)

loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

image = Image.open('images/pcnn_lr:0.00020_nr-resnet3_nr-filters80_0.png')
x = TF.to_tensor(image)
x = x[:1]
x.unsqueeze_(0)
print(x.shape)

output = loaded_model(x)
prediction = torch.argmax(output)
print(output)

# print(loaded_model.state_dict())

