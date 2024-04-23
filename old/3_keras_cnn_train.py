'''
Author: BNDou
Date: 2024-04-22 23:24:04
LastEditTime: 2024-04-24 03:03:08
FilePath: \Captchas_BOC\old\3_keras_cnn_train.py
Description: 
'''
import os
import random
import cv2
from keras.layers import Flatten, Input, Dropout, Conv2D, MaxPooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import LabelBinarizer


def create_model(input_size, class_num):
    '''
    卷积神经网络
    '''
    # 输入层
    input = Input(shape=input_size)
    # 卷积层，输出维度16，卷积核大小3x3，激活函数relu，填充方式same
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input)
    # 最大池化层，输出维度的一半，池化核大小2x2，步长2x2
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # 卷积层，输出维度64，卷积核大小3x3，激活函数relu，填充方式same
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    # 最大池化层，输出维度的一半，池化核大小2x2，步长2x2
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # 卷积层，输出维度256，卷积核大小3x3，激活函数relu，填充方式same
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(input)
    # 最大池化层，输出维度的一半，池化核大小2x2，步长2x2
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # 展平层，方便全连接层处理
    x = Flatten()(x)
    # 全连接层，输出维度1024，激活函数relu
    x = Dense(1024, activation='relu')(x)
    # Dropout层，以0.5的概率将输入层nect_layer的输出值置为0，防止过拟合
    x = Dropout(0.5)(x)
    # 全连接层，输出维度2048，激活函数relu
    x = Dense(2048, activation='relu')(x)
    # Dropout层，以0.5的概率将输入层nect_layer的输出值置为0，防止过拟合
    x = Dropout(0.5)(x)
    # 全连接层，输出维度class_num，激活函数softmax
    x = Dense(class_num, activation='softmax')(x)
    # 返回模型
    model = Model(inputs=input, outputs=x)
    # 配置模型，使用Adam优化器，分类任务使用交叉熵损失函数，评估准确率
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


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
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image, (16, 16))
        image = np.expand_dims(image, axis=-1)
        data.append(image)
        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # 对图像数据做scale操作
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # 数据集切分
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels,
                                                      test_size=0.25,
                                                      random_state=42)

    # 转换标签为one-hot encoding格式
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    return trainX, trainY, testX, testY


if __name__ == '__main__':
    (trainX, trainY, testX, testY) = load_data()
    print("------准备训练网络------")
    # 建立卷积神经网络
    model = create_model(input_size=(16, 16, 1), class_num=26)
    H = model.fit(trainX,
                  trainY,
                  validation_data=(testX, testY),
                  epochs=50,
                  batch_size=16)
