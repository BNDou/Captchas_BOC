'''
Author: BNDou
Date: 2024-04-25 03:54:41
LastEditTime: 2024-04-26 05:07:41
FilePath: \Captchas_BOC\test.py
Description: 
'''
import os
import pickle
import cv2
from keras.models import load_model
from keras_ocr_api import recognize_capchar

# 模型和标签的访问路径
MODEL_PATH = r'.\model\keras_model.h5'
LABEL_PATH = r'.\model\keras_lb.pickle'
# 测试图片的访问路径
IMAGE_PATH = r'.\demo\captcha_images'

if __name__ == '__main__':
    # 读取模型和标签
    print("------读取模型和标签------")
    model = load_model(MODEL_PATH)
    lb = pickle.loads(open(LABEL_PATH, "rb").read())

    # 读取测试图片的路径
    print("------读取测试图片的路径------")
    imagePaths = []
    # 将每个图片的路径添加到imagePaths列表中
    for label in os.listdir(IMAGE_PATH):
        imagePaths.append(os.path.join(IMAGE_PATH, label))

    # 遍历每个图片的路径
    print("------开始识别------")
    for imagePath in imagePaths:
        char_key = recognize_capchar(imagePath, model, lb)
        print(f"{imagePath} 识别结果：{char_key}")
