'''
Author: BNDou
Date: 2024-07-06 17:00:15
LastEditTime: 2024-07-06 21:36:29
FilePath: \Captchas_BOC\e_api.py
Description: 
'''
import base64
import os
import time
import requests

# 测试图片的访问路径
IMAGE_PATH = r'.\demo\captcha_images'


def identify_GeneralCAPTCHA(code):
    """
    识别验证码
    :param code: 图片路径或base64编码的字符串
    :return: 识别结果
    """
    url = "http://bndou.top:6688/api/ocr/image"  # 请替换为实际的API URL
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "img_base64": str(code),
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()["result"].upper()
        # print("识别结果：", result)
        return result
    else:
        print("识别失败")


def image_to_base64(image_path):
    """
    将图片转换为base64编码
    :param image_path: 图片路径
    :return: base64编码的字符串
    """
    with open(image_path, 'rb') as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")
    return base64_data


if __name__ == '__main__':
    # 读取测试图片的路径
    print("------读取测试图片的路径------")
    imagePaths = []
    # 将每个图片的路径添加到imagePaths列表中
    for label in os.listdir(IMAGE_PATH):
        imagePaths.append(os.path.join(IMAGE_PATH, label))
    imagePaths = sorted(imagePaths)

    # 遍历每个图片的路径
    print("------开始识别------")
    for imagePath in imagePaths:
        code = image_to_base64(imagePath)
        char_key = identify_GeneralCAPTCHA(code)
        print(f"{imagePath}  识别结果：{char_key}")
        # time.sleep(3)

    # # 调用示例
    # code = image_to_base64("./temp/temp1.jpg")
    # result = identify_GeneralCAPTCHA(code)
