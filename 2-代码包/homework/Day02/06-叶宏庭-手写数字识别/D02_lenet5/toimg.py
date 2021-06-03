import torch
import torch.nn as nn
import torch.nn.functional as fu
import torchvision
from  torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import numpy as np
import cv2

transform = Compose([ToTensor(), ]) 
train_mnist = MNIST(root="datasets", train=True, download=True, transform=transform)

img, target = train_mnist[5]

# 1. *255
img = img.mul(255)
# 2. 转换像素的基本表示类型uint8
img = img.numpy().copy()
img = img.astype(np.uint8)
# 图像格式问题
img = img.transpose(1, 2, 0)
print(img.shape)
# 3. 保存
cv2.imwrite("digit1.jpg",img)
