import torch
import torch.nn as nn
import torch.nn.functional as fu
import torchvision
from  torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import numpy as np
import cv2

class LetNet5(nn.Module):

    # 构造器（定义使用卷积运算）
    def __init__(self, class_num=10):
        super(LetNet5, self).__init__()    # 注意不要丢掉
        # 定义需要使用的运算
        # self.conv1 = nn.Conv2d(                  # 1 * 32 * 32 -> 6 * 28 * 28  -> 6 * 14 * 14 
        #     1,  # 输入图像的通道
        #     6,  # 输出的特征图个数（输出通道）
        #     5)  # 卷积核的大小
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 1 * 28 * 28 -> 6 * 28 * 28  -> 6 * 14 * 14
        self.conv2 = nn.Conv2d(6, 16, 5)         # 6 * 14 * 14  -> 16 * 10 * 10 -> 16 * 5 * 5
        self.conv3 = nn.Conv2d(16, 120, 5)       # 16 * 5 * 5 -> 120 * 1 * 1 - >120向量

        # 分类器
        self.fc1 = nn.Linear(120, 84)            # 120 -> 84
        self.fc2 = nn.Linear(84, 10)             # 84 -> 10



    # 利用定义的运算实现图像到向量的运算过程
    def forward(self, x):  # 输入的图像 1 * 32 * 32
        """
            x的有固定的格式：4维：（Number，Channel，High，Width）：NCHW
        """
        x = self.conv1(x)            # 1 * 32 * 32 -> 6 * 28 * 28
        x = fu.max_pool2d(x, (2, 2)) # 6 * 28 * 28  -> 6 * 14 * 14 
        x = fu.relu(x)               # 过滤负值 6 * 14 * 14  -> 6 * 14 * 14

        x = self.conv2(x)            # 6 * 14 * 14 -> 16 * 10 * 10
        x = fu.max_pool2d(x, (2, 2)) # 16 * 10 * 10  -> 16 * 5 * 5 
        x = fu.relu(x)               # 过滤负值 16 * 5 * 5  -> 16 * 5 * 5

        x = self.conv3(x)            # 16 * 5 * 5 -> 120 * 1 * 1
        x = fu.relu(x)               # 过滤负值 120 * 1 * 1  -> 120 * 1 * 1

        # 格式转换
        x = x.view(-1, 120)

        # 分类器
        x = self.fc1(x)
        x = fu.relu(x)

        x = self.fc2(x)
        return x

# 识别
# 创建模型
model = LetNet5()
# 加载预训练模型
state = torch.load("./lenet.pt")
model.load_state_dict(state)
# # ---------------------------------数据集测试
# transform = Compose([ToTensor(), ]) 
# valid_mnist = MNIST(root="datasets", train=False, download=True, transform=transform)

# for i in range(10):
#     img, label = valid_mnist[i]
#     # print(type(img))   # NCHW
#     # print(img.shape)
#     img = torch.unsqueeze(img, 0)
#     # print(img.shape)
#     y_pred = model(img)
#     # print(y_pred)
#     prob = fu.softmax(y_pred, dim=1)
#     # print(prob)
#     pred_cls = torch.argmax(prob, dim=1)
#     print(pred_cls.detach().numpy()[0],"->",label)
# ---------------------------------图片测试
# 1. 读取图片
img_digit = cv2.imread("digit_02.png")
# 2. 图像的预处理
img_digit = cv2.cvtColor(img_digit, cv2.COLOR_BGR2GRAY)
img_digit = img_digit.astype("float32")
img_digit = img_digit/255.0
# print(img_digit)
# 3. 转换Tensor
img_tensor = torch.from_numpy(img_digit).clone()

y_digit = model(img_tensor.view(1, 1, 28, 28))
prob_digit = fu.softmax(y_digit, dim=1)
result = torch.argmax(prob_digit)
print(F"预测结果是：{result.detach().numpy()}")
