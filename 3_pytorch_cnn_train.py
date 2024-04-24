'''
Author: BNDou
Date: 2024-04-22 23:24:04
LastEditTime: 2024-04-24 14:51:54
FilePath: \Captchas_BOC\3_pytorch_cnn_train.py
Description: 
'''
import os
import random
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision

EPOCH = 50
BATCH_SIZE = 16
LR = 0.001
DOWNLOAD_MNIST = False


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding='same',
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.out1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.out2 = nn.Sequential(
            nn.Linear(1024, 2048, True),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.out3 = nn.Sequential(nn.Linear(2048, 26, True), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)  # 卷积层
        x = self.conv2(x)  # 卷积层
        x = self.conv3(x)  # 卷积层
        x = nn.Flatten()(x)  # 展平层
        output = self.out1(x)  # 全连接层
        output = self.out2(x)  # 全连接层
        output = self.out3(x)  # 全连接层
        return output


def load_data():
    image_path = './chars_dict'
    data = []
    labels = []
    imagePaths = []

    # 遍历image_path下的所有文件夹
    for label in os.listdir(image_path):
        # 遍历每个文件夹下的所有图片
        for image in os.listdir(os.path.join(image_path, label)):
            # 将每个图片的路径添加到imagePaths列表中
            imagePaths.append(os.path.join(image_path, label, image))

    # 拿到图像数据路径，方便后续读取
    imagePaths = sorted(imagePaths)
    # 随机打乱图像数据路径
    random.seed(42)
    random.shuffle(imagePaths)

    # 遍历读取数据
    for imagePath in imagePaths:
        # 读取图像数据
        image = Image.open(imagePath)
        image = image.resize((16, 16))
        # plt.imshow(image)
        # plt.show()
        # 读取图像数据并转换为tensor格式
        image = torchvision.transforms.ToTensor()(image).float() / 255.0
        data.append(image)

        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # 数据集切分
    dataset_length = len(data)
    train_size = int(0.8 * dataset_length)
    train_data, test_data = data[:train_size], data[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]
    train_data, test_data = torch.stack(train_data), torch.stack(test_data)

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    # 加载数据集
    print("------加载数据集------")
    train_x, train_y, test_x, test_y = load_data()

    # 建立卷积神经网络
    print("------建立卷积神经网络------")
    cnn = CNN()
    print(cnn)

    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    # 训练网络
    print("------开始训练网络------")
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(zip(train_x, train_y)):
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float(
                    (pred_y == test_y.data.numpy()).astype(int).sum()) / float(
                        test_y.size(0))
                print('Epoch: ', epoch,
                      '| train loss: %.4f' % loss.data.numpy(),
                      '| test accuracy: %.2f' % accuracy)

    # 测试网络：10个数据打印预测
    print("------开始测试网络------")
    test_output, _ = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print('预测：', pred_y, '实际：', test_y[:10].numpy())
