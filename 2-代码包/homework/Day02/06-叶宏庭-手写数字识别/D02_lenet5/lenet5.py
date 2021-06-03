import torch
import torch.nn as nn
import torch.nn.functional as fu
import torchvision
from  torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import cv2
import datetime

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



# 数据集
# 1. 提供转换函数
transform = Compose([ToTensor(), ])  # 图像的预处理
# 2. 加载数据集
train_mnist = MNIST(root="datasets", train=True, download=True, transform=transform)
valid_mnist = MNIST(root="datasets", train=False, download=True, transform=transform)
# 3. 批次加载器
train_loader = DataLoader(dataset=train_mnist, shuffle=True, batch_size=1000)
valid_loader = DataLoader(dataset=valid_mnist, shuffle=True, batch_size=1000)

net = LetNet5()

# 开始训练
# 准备
epoch = 3
learning_rate = 0.01

# 损失函数
f_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 权重迭代更新器


for e in range(epoch):
    now_time = datetime.datetime.now()
    n = 0
    for x_, y_ in train_loader:
        # 向前预测结果
        y_pred = net.forward(x_.view(-1, 1, 28, 28))
        # 使用结果计算误差
        loss = f_loss(y_pred, y_)
        # 根据误差求导
        optimizer.zero_grad()
        loss.backward()
        # 使用优化器迭代权重
        optimizer.step()
        n += 1
    with torch.no_grad():
        # 损失值输出
        # print("轮数{e:03d}/批次{n:02d}, 损失值：{loss:6.4f}")
        # 验证
        all_num = 0.0   # 测试集的总样本数
        acc = 0.0     # 识别正确个数
        for v_x, v_y in valid_loader:
            all_num += len(v_y)  # 统计样本总数
            v_pred = net.forward(v_x.view(-1, 1, 28, 28))
            prob = fu.softmax(v_pred, dim=1)
            cls_pred = torch.argmax(prob, dim=1)

            # 判别预测类别是否与标签相等
            acc += (cls_pred == v_y).float().sum()
        print(F"轮数{e:03d}, \t损失值：{loss:6.4f}, \t识别正确率：{100.0 * acc/all_num:6.2f}%, \t训练用时：{(datetime.datetime.now()-now_time).seconds}s")


# 模型的保存
torch.save(net.state_dict(), "./lenet.pt")


