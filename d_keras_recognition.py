'''
Author: BNDou
Date: 2024-04-23 01:00:34
LastEditTime: 2024-04-30 06:25:24
FilePath: \Captchas_BOC\d_keras_recognition.py
Description: 
    使用Keras模型进行验证码识别
'''
import os
import cv2
import numpy as np
import pickle
from keras.models import load_model


# 模型和标签的访问路径
MODEL_PATH = '.\model\keras_model.h5'
LABEL_PATH = '.\model\keras_lb.pickle'
# 测试图片的访问路径
IMAGE_PATH = '.\chars_dict'


if __name__ == '__main__':
    # 读取模型和标签
    print("------读取模型和标签------")
    model = load_model(MODEL_PATH)
    lb = pickle.loads(open(LABEL_PATH, "rb").read())

    # 加载测试数据并进行相同预处理操作
    imagePaths = []
    # 遍历image_path下的所有文件夹
    for label in os.listdir(IMAGE_PATH):
        # 遍历每个文件夹下的所有图片
        for image in os.listdir(os.path.join(IMAGE_PATH, label)):
            # 将每个图片的路径添加到imagePaths列表中
            imagePaths.append(os.path.join(IMAGE_PATH, label, image))

    # 预测失败集合
    err_list = []
    # 遍历每个图片
    # for imagePath in random.sample(imagePaths, 50): # 随机选择50个图片进行测试
    for imagePath in imagePaths: # 遍历所有图片
        # 读取标签
        label = imagePath.split(os.path.sep)[-2]
        # 读取图片
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image, (16, 16))
        # scale图像数据
        image = image.astype("float") / 255.0
        image = np.expand_dims(image, axis=-1)
        # 对图像进行拉平操作
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # 预测
        preds = model.predict(image)
        # 得到预测结果以及其对应的标签
        i = preds.argmax(axis=1)[0]
        # 在图像中把结果画出来
        log = f"实际：{label}  预测：{lb[i]}  验证：{'⭕' if label == lb[i] else '❌'}"
        print(log)
        if label != lb[i]:
            err_list.append(log)

    print("------预测报告------")
    print(f"本次预测成功率：{100 - len(err_list) / len(imagePaths) * 100}%\n数据总数：{len(imagePaths)}\n预测失败数：{len(err_list)}\n预测失败率：{len(err_list) / len(imagePaths) * 100}")
    for err in err_list:
        print(err)

