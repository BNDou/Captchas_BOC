'''
Author: BNDou
Date: 2024-04-25 01:54:43
LastEditTime: 2024-04-26 05:12:43
FilePath: \Captchas_BOC\keras_ocr_api.py
Description: 
    使用Keras OCR API识别验证码
'''

import pickle
import cv2
from keras.models import load_model
import numpy as np
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

# 确保此路径正确指向 msedgedriver.exe
driver_path = r'msedgedriver.exe'
# 设置 Edge 以无头模式运行
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
# 创建 Service 对象，并确保 executable_path 指向 msedgedriver.exe
service = Service(executable_path=driver_path)
# 初始化 WebDriver
driver = webdriver.Edge(service=service, options=options)

# 模型和标签的访问路径
MODEL_PATH = '.\model\keras_model.h5'
LABEL_PATH = '.\model\keras_lb.pickle'
# 读取模型和标签
print("------读取模型和标签------")
MODEL = load_model(MODEL_PATH)
LB = pickle.loads(open(LABEL_PATH, "rb").read())


def get_gif_first_frame(gif_path):
    '''
    这个函数用于从给定的gif文件中提取第一帧图片。
    :param gif_path: gif文件的路径
    :return: 返回第一帧图片
    '''
    gif = cv2.VideoCapture(gif_path)
    _, frame = gif.read()
    gif.release()
    return frame


def get_cutted_patches(capchar):
    '''
    这个函数用于读取给定的验证码图片文件，并切割出字符图像。
    :param image: 验证码图片
    :return: 返回切割后的字符图像列表
    '''
    # 读取图片
    img = cv2.imread(capchar)
    if img is None:
        img = get_gif_first_frame(capchar)
    # img = img[1:-1, 1:-1]

    # 将图像二值化
    # 127和128的像素是干扰，都变为255 即白色，其余非白色变黑 即字符
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            b, g, r = img[y, x]
            if (b, g, r) not in [(128, 128, 128), (127, 127, 127),
                                 (255, 255, 255)]:
                img[y, x] = (255, 255, 255)
            else:
                img[y, x] = (0, 0, 0)
    # cv2.imshow('thresh', img)
    # cv2.waitKey(0)

    # 膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 矩形结构
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # dilation = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
    dilation = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
    # cv2.imshow('dilation', dilation)
    # cv2.waitKey(0)

    # 获取轮廓
    dilation = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # # 显示轮廓
    # drawContours = cv2.drawContours(cv2.imread(capchar), contours, -1,
    #                                 (0, 0, 255), 2)
    # cv2.imshow('drawContours', drawContours)
    # cv2.waitKey(0)

    # 只保留面积最大的6个轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    # 获取所有外接矩形
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # 排序
    (contours, boundingBoxes) = zip(
        *sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0]))

    # 切割出字符图像
    # 初始化一个空列表，用于存储切割后的字符图像
    cut_chars = []
    for box in boundingBoxes:
        x, y, w, h = box
        char = dilation[y:y + h, x:x + w]
        cut_chars.append(char)
        cv2.imshow('char', char)
        cv2.waitKey(0)

    return cut_chars


def recognize_capchar(capchar, model, lb):
    '''
    这个函数用于识别给定的验证码图片。
    :param path: 验证码图片的路径
    :param model: 训练好的模型
    :param lb: 标签
    :return: 返回识别结果
    '''
    chars_key = ""
    cut_chars = get_cutted_patches(capchar)

    for char in cut_chars:
        img = cv2.resize(char, (16, 16))
        img = img.astype("float") / 255.0
        img = np.expand_dims(img, axis=-1)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # 预测
        preds = model.predict(img)
        # 获取预测结果以及其对应的标签
        i = preds.argmax(axis=1)[0]
        chars_key += lb[i]

    return chars_key


if __name__ == '__main__':
    url = 'https://cdn1.cmcoins.boc.cn/ocas-web/ImageValidation/validation1704292251688.gif'
    driver.get(url)
    # 先找到验证码对应的网页元素
    ele_piccaptcha = driver.find_element(By.XPATH, '/html/body/img')
    # 然后直接调用这个元素的screenshot方法，参数是保存的路径即可实现截图
    ele_piccaptcha.screenshot('./temp/temp_capchar.jpg')
    # 然后调用识别函数
    chars_key = recognize_capchar('./temp/temp_capchar.jpg', MODEL, LB)
    print(f'识别结果为：{chars_key}')
